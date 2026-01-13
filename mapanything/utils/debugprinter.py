import torch
from typing import Any

class DebugPrinter:
    def __init__(self, max_depth=12, indent_size=2, line_char="="):
        self.max_depth = max_depth
        self.indent_size = indent_size
        self.line_char = line_char
        self._visited = set()

    def reset(self):
        """Reset visited set for each print session."""
        self._visited = set()

    # ============================================================
    # PUBLIC API
    # ============================================================
    def print(self, obj: Any, name: str = "root"):
        """Public method: print a variable with a clean section header."""
        self.reset()
        header = f"[DEBUG] {name}"
        print("\n" + header)
        print(self.line_char * len(header))

        self._print_recursive(obj, name, depth=0)

        print(self.line_char * len(header) + "\n")

    # ============================================================
    # INTERNAL RECURSIVE LOGIC
    # ============================================================
    def _print_recursive(self, obj: Any, name: str, depth: int):
        indent = " " * (depth * self.indent_size)

        # depth limit
        if depth > self.max_depth:
            print(f"{indent}{name}: <max_depth_reached>")
            return

        # recursion protection
        obj_id = id(obj)
        if obj_id in self._visited:
            print(f"{indent}{name}: <recursive_ref>")
            return
        self._visited.add(obj_id)

        # TYPE DISPATCH
        if torch.is_tensor(obj):
            self._handle_tensor(obj, name, depth)
            return

        if obj is None:
            self._handle_none(obj, name, depth)
            return

        if isinstance(obj, (int, float, bool, str)):
            self._handle_scalar(obj, name, depth)
            return

        if isinstance(obj, (list, tuple, set)):
            self._handle_sequence(obj, name, depth)
            return

        if isinstance(obj, dict):
            self._handle_dict(obj, name, depth)
            return

        if hasattr(obj, "__dict__"):
            self._handle_object(obj, name, depth)
            return

        self._handle_unknown(obj, name, depth)

    # ============================================================
    # TYPE HANDLERS
    # ============================================================
    def _handle_tensor(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        print(f"{indent}{name}: Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})")

    def _handle_none(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        print(f"{indent}{name}: None")

    def _handle_scalar(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        print(f"{indent}{name}: {obj} ({type(obj).__name__})")

    def _handle_sequence(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        typename = type(obj).__name__
        print(f"{indent}{name}: {typename}(len={len(obj)})")

        for i, elem in enumerate(obj):
            self._print_recursive(elem, f"{name}[{i}]", depth + 1)

    def _handle_dict(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        keys = list(obj.keys())
        print(f"{indent}{name}: dict(keys={keys})")

        for k, v in obj.items():
            self._print_recursive(v, f"{name}.{k}", depth + 1)

    def _handle_object(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        cls_name = obj.__class__.__name__
        print(f"{indent}{name}: {cls_name}")

        for k, v in obj.__dict__.items():
            self._print_recursive(v, f"{name}.{k}", depth + 1)

    def _handle_unknown(self, obj, name, depth):
        indent = " " * (depth * self.indent_size)
        print(f"{indent}{name}: <{type(obj).__name__}> (unhandled type)")
