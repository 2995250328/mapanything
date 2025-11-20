# `_loss_fn` 损失函数详解

## 1. 整体结构与输入输出

这个 `_loss_fn` 是一个**统一的几何回归损失模块**，主要服务于“预测 3D 坐标 / 场景坐标 + 相机几何”的任务。它同时兼容两类回归头与两种监督模式：

- 回归头：
  - FiLM 头：直接输出带全局尺度的 3D 坐标和一个不确定性标量；
  - Decoupled+Scale 头：输出无尺度坐标（unit XYZ）和独立的尺度标量 scale。
- 监督模式：
  - `mode = "xyz"`：在世界坐标系下，直接对 3D 坐标做监督（场景坐标回归）；
  - `mode = "reproj"`：通过相机位姿和内参把 3D 点投影到像素平面，对 2D 重投影误差做监督。

函数输入包括：预测 `preds`（可能带 scale）、batch 中的相机参数和监督数据、一个外部的重投影损失对象 `repro_loss`、当前 `global_step`、以及控制模式与超参的 `loss_cfg`。输出是 `(total_loss, metrics)`，其中 `metrics` 会记录平均误差、平均不确定性、尺度正则项等，用于日志和监控。

---

## 2. 预测解包与不确定性建模

首先，函数对 `preds` 的多种形态做统一解包：如果是纯张量 `Tensor[N,4]`，则默认其含义为 `[X,Y,Z,raw]`，对应 FiLM 头；如果是 `(Tensor[N,4], scale[N])` 或列表，则认为是解耦头：`preds[:, :3]` 是 unit 坐标，`scale` 是额外的尺度；如果是 dict，则从 `preds["preds"]` 中拿坐标和 raw，并从 `preds["scale"]` 中拿尺度。若预测仍然是 `[B,4,H,W]` 的特征图形式，会先展平为 `[N,4]`，保证后续以“点”为单位进行计算。

随后拆出 `coords_pred_in = preds[:, :3]` 和 `raw = preds[:, 3]`。`coords_pred_in` 要么已经是带尺度的 XYZ，要么是 unit XYZ。`raw` 用来建模每个点的预测不确定性，并根据 `loss_cfg.conf_mode` 映射到正数 σ。如果使用 `"confidence"` 模式，则把 `raw` 经过 `sigmoid` 得到置信度 `p∈(0,1)`，再在 `[sigma_max, sigma_min]` 区间线性插值得到 `sigma`，置信度越高 sigma 越小，对应更严格的惩罚；如果使用 `"log_sigma"`（默认），则通过 `softplus(raw) + eps` 得到正数 sigma，并截断到 `[sigma_min, sigma_max]`，然后取对数 `log_sigma`。这种设计是典型的异方差回归设置，允许网络为不同样本分配不同的权重。

在尺度处理方面，如果存在 `scale`，则认为 `coords_pred_in` 是无尺度坐标 `coords_unit`，真正用于监督的有尺度坐标为 `coords = coords_unit * scale`；同时保留 `coords_unit`，后续可以对无尺度深度做 unit 正则。如果 `scale` 为 `None`，则直接认为 `coords_pred_in` 已经是带尺度的 `coords`，并将 `coords_unit` 置为 `None`。此外，内部定义了一个 `_invert_c2w_to_w2c` 函数，用于将 batch 中的 `c2w`（支持 `[N,4,4]` 和 `[N,3,4]` 两种格式）转为世界到相机的 `R,t`，方便后续从世界坐标转换到相机坐标系。

---

## 3. xyz 模式：3D 世界坐标异方差回归

当 `loss_cfg.mode = "xyz"` 时，主监督目标是 batch 中的 `target_world[N,3]`。首先计算预测世界坐标 `coords` 与 GT 世界坐标 `target_world` 的差值，得到每个点的三维误差向量，再对每个样本的误差向量求 L2 范数，得到标量误差 `err`。随后使用之前得到的 `sigma` 和 `log_sigma` 构造每个点的异方差损失，一般形式为 `L_i = log_sigma_i + sqrt(2) * (err_i / sigma_i)`，最后对所有样本求均值得到总损失 `total`。这个形式对应 Laplace 分布下的异方差负对数似然：如果某个点误差较大，网络可以通过预测更大的 sigma 来降低它在损失中的权重，但 sigma 增大本身又会提高 `log_sigma` 项，实现对“放弃困难样本”的抑制，从而让 loss 对噪声、遮挡、模糊等情况更鲁棒。

在 xyz 模式中还可以叠加尺度正则（`scale_reg`）。如果 `scale_reg.enabled` 为真且 batch 中有 `c2w`，函数会利用 `c2w` 反转得到 `R,t`，然后根据 `scale_reg.variant` 的不同，对不同的深度分布施加约束。若使用 `variant = "unit"` 并且存在 `coords_unit`，则说明当前使用的是“unit + scale”的解耦头，此时会将 `coords_unit` 变换到相机坐标系，提取 Z 分量得到无尺度深度 `Zu`，并对其中位数 `median(Zu)` 做 unit 正则（期望其接近 1），形式是对 `log(median(Zu))` 做 L1 惩罚。若不是这种情况，则对有尺度的 `coords` 做变换，得到深度 `Zp`，然后：`variant = "match_to_gt"` 时，将 `Zp` 的中位数与由 GT 世界坐标 + 相机位姿计算出来的 GT 深度中位数进行 log 空间的比值约束；`variant = "prior"` 时，则把 `median(Zp)` 拉向配置中的 `depth_prior`。这些正则统一通过一个 `_scale_reg_add` 函数实现，返回更新后的 `total` 和当前正则项的数值。xyz 模式的 `metrics` 中会记录平均 3D 误差、平均 σ、总体 loss、模式标记 `"xyz-sparse"`，以及（若启用）尺度正则项数值，便于在训练中追踪尺度行为。

---

## 4. reproj 模式：重投影误差与无效点处理

当 `loss_cfg.mode = "reproj"` 时，主监督转为 2D 像素级重投影误差。函数首先从 batch 中取出相机内参矩阵 `K`、外参 `c2w` 和监督像素坐标 `pixels[N,2]`，再用 `_invert_c2w_to_w2c` 得到世界到相机的 `R,t`。然后将预测世界坐标 `coords` 扩展为 `[N,3,1]`，通过 `Xc = R @ Xw + t` 得到相机坐标系下的点 `Xc`，其 Z 分量即深度 `z`，再对其进行 `clamp_min(depth_min)` 得到 `z_clamped` 和一维深度向量 `z_flat`。接下来通过 `uvh = K @ Xc`、`uv = uvh[:,:2] / z_clamped` 得到预测像素坐标 `uv`，和监督像素坐标 `px` 求 L1 距离得到每个点的重投影误差 `repro_err`。

reproj 模式的一个关键点在于**区分“有效点”和“无效点”**。无效点的判定条件包括：深度小于 `depth_min`（点在相机后方或过近）、深度大于 `depth_max`（点过远）、重投影误差大于 `repro_loss_hard_clamp`（像素误差极端）。这些条件按位 OR 得到 `invalid_mask`，其反集为 `valid_mask`。对于有效点，调用外部传入的 `repro_loss.compute(repro_err[valid_mask], global_step)` 计算主损失，这个 `repro_loss` 可以是简单的 L1、L2，也可以是 Huber 或带 curriculum 的 schedule。对于无效点，不直接用巨大的 `repro_err`，而是通过几何方式构造一个“期望相机位置”：先得到内参逆 `invK`（若 batch 中有 `intrinsics_inv` 则直接使用，否则对 `K` 求逆），把监督像素 `(u,v)` 拼成 `(u,v,1)` 的列向量，乘以 `invK` 得到归一化射线方向，再乘以配置中的 `depth_target` 得到目标相机坐标 `Xc_tgt`，最终用 `(Xc_tgt - Xc)` 的 L1 误差来约束这些无效点。这等价于说：对那些坐标明显不合理的预测，不让它们通过巨大 reprojection loss 主导训练，而是把它们往“正确像素方向、合理深度”的位置拉回。总损失为 `(loss_valid + loss_invalid) / N`，平均到所有样本上。

在 reproj 模式下同样可以叠加尺度正则。缺省情况下 `scale_reg.variant` 可能是 `"prior"`，此时对 `z_flat` 做 median-based 正则，让预测深度分布的中位数靠近 `depth_prior`。如果设置为 `"match_to_gt"` 且 batch 中提供了 `target_world`，则可以借助 GT 世界坐标和相机姿态计算出 GT 深度中位数，并对预测深度中位数与 GT 的比值做 log-L1 约束。若使用 `"unit"` 且存在 `coords_unit`，则会将 `coords_unit` 通过相机变换得到无尺度深度 `Zu`，再对 `median(Zu)` 做 unit 正则。reproj 模式的 `metrics` 会记录平均像素误差（`err_mean_px`）、平均 σ、总损失、模式标签 `"reproj-sparse"`，以及（若启用）尺度正则值。

---

## 5. 尺度正则 scale_reg 的统一视角

`scale_reg` 是这个损失函数在设计上的一个关键点，它用一个统一的形式对整体尺度进行柔和约束：无论是 match_to_gt、unit 还是 prior，本质上都在对“深度分布的中位数”做 log 空间的 L1 正则。具体地说，令 `med_pred = median(z_pred)`，参考值 `ref` 可以是 GT 深度中位数、1 或 `depth_prior`，正则项基本都是 `weight * |log((med_pred+eps)/(ref+eps))|`。这种形式的优点在于：使用 median 而不是 mean，使得正则对 outlier 不敏感；在 log 空间约束比值，实际是在约束“尺度因子”而非绝对差，更符合多视图几何中“整体尺度不定”的性质；配合一个较小的 `weight`（例如 1e-3 量级），可以在不破坏局部几何学习的前提下，给出一个全局的尺度锚点。

在使用解耦头时，`coords_unit` 和 `scale` 的分工更加清晰：unit 坐标主要负责方向和相对几何结构，scale 则负责整体大小。此时可以通过 `variant="unit"`，对“unit 坐标 + 相机位姿得到的深度”的中位数施加 unit 正则，让 unit 部分的尺度分布稳定在一个合理区间，而整体场景尺度则由 scale 来调整；如果没有解耦，FiLM 头也可以通过 `match_to_gt` 或 `prior` 将预测的场景尺度在统计意义上对齐到 GT 或先验，并由异方差项自动调整局部点的权重。

---

## 6. 总结与使用建议

综合来看，`_loss_fn` 可以被看作一个“面向几何坐标回归”的统一训练接口：它在输入端通过对 `preds` 的容错解包，适配了 FiLM 与 unit+scale 两种回归头；在监督端通过 `mode="xyz"` 和 `mode="reproj"` 兼容了世界坐标监督与重投影监督；内部通过 `raw → sigma` 实现了 per-point 的异方差回归，使得模型可以自动降低对困难点、噪声点的权重；再通过 median-based 的 `scale_reg` 对整体深度尺度进行轻量锚定，避免训练过程中尺度飘飞；在 reproj 模式中，还专门对深度极端或像素误差极大的“无效点”做了几何矫正，而不是让其爆炸性误差主导优化。实际使用时，如果你有较完整的 3D 场景坐标 GT，可以优先使用 xyz 模式并配合 `"log_sigma"` 的异方差项；如果只有 2D 像素监督和相机参数，则可以使用 reproj 模式，并根据场景设置合理的 `repro_loss_hard_clamp`、`depth_target` 和 `scale_reg` 变体。通过统一的这一套 `_loss_fn`，你可以在同一代码框架下方便地做 FiLM vs 解耦、xyz vs reproj、不同 scale_reg 变体的对比实验。
