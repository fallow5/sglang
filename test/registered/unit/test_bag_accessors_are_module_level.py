"""The config-bag accessors are module functions of no arguments, and a module
binds each name once.

`get_disagg()` and its siblings answer for the whole process. Three ways a
config sweep breaks that, each of which imports fine and fails only when the
path is actually run:

* calling one as `something.get_disagg()` -- it reads as "this object's
  disaggregation config", which a rewrite of `x.server_args.field` produces;
* passing arguments to one, which happens when the accessor name collides with
  a same-named function the module already imported (`model_loader.get_model`
  vs the model bag) and the collision silently wins;
* binding an accessor name twice in a module, which is that collision itself.

The three are checked together because they are one invariant: the name means
the process-wide bag, takes nothing, and is spelled the same everywhere.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import sglang

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

_BAG_ACCESSORS = frozenset(
    {
        "get_exec",
        "get_memory",
        "get_schedule",
        "get_model",
        "get_spec",
        "get_serving",
        "get_observability",
        "get_disagg",
        "get_lora",
        "get_mm",
        "get_device",
        "get_parallel",
    }
)


def _accessor_bindings(tree):
    """(local name -> accessor, module aliases, other bindings of those names).

    Aliases count: `get_parallel as _gp` is still the process accessor, and a
    module scoped only to literal names would skip the file that uses it.
    """
    local_to_accessor, module_aliases, rebound = {}, set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            in_runtime_context = bool(node.module) and node.module.endswith(
                "runtime_context"
            )
            for a in node.names:
                local = a.asname or a.name
                if in_runtime_context and a.name in _BAG_ACCESSORS:
                    local_to_accessor[local] = a.name
                elif local in _BAG_ACCESSORS:
                    rebound.setdefault(local, []).append(node.lineno)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("runtime_context"):
                    module_aliases.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _BAG_ACCESSORS:
                rebound.setdefault(node.name, []).append(node.lineno)
    return local_to_accessor, module_aliases, rebound


class TestBagAccessorsAreCalledAsFunctions(CustomTestCase):
    def _scan(self):
        member_calls, arg_calls, collisions = [], [], []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            local_to_accessor, module_aliases, rebound = _accessor_bindings(tree)
            if not local_to_accessor and not module_aliases:
                continue
            where = path.relative_to(_PACKAGE_ROOT.parent)
            for name, lines in rebound.items():
                if name in local_to_accessor:
                    collisions.append(
                        f"{where}:{lines[0]} binds {name!r} twice; the bag "
                        "accessor and something else share the name"
                    )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_accessor = False
                if isinstance(func, ast.Name) and func.id in local_to_accessor:
                    is_accessor = True
                elif isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id in module_aliases
                        and func.attr in _BAG_ACCESSORS
                    ):
                        is_accessor = True
                    elif func.attr in local_to_accessor and not (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "self"
                        # A class may own a same-named member; the sweep's
                        # mistake is calling it on a config-carrying object.
                    ):
                        member_calls.append(
                            f"{where}:{node.lineno} calls .{func.attr}() on an object"
                        )
                if is_accessor and (node.args or node.keywords):
                    arg_calls.append(
                        f"{where}:{node.lineno} passes arguments to "
                        f"{ast.unparse(func)}(), which answers for the process "
                        "and takes none"
                    )
        return member_calls, arg_calls, collisions

    def test_no_accessor_is_called_on_an_object(self):
        member_calls, _, _ = self._scan()
        self.assertEqual(
            member_calls,
            [],
            "config-bag accessors answer for the process, not for an object:\n  "
            + "\n  ".join(member_calls),
        )

    def test_no_accessor_is_called_with_arguments(self):
        _, arg_calls, _ = self._scan()
        self.assertEqual(
            arg_calls,
            [],
            "an accessor called with arguments is a name collision that the "
            "import order decided:\n  " + "\n  ".join(arg_calls),
        )

    def test_no_module_binds_an_accessor_name_twice(self):
        _, _, collisions = self._scan()
        self.assertEqual(
            collisions,
            [],
            "the later import wins and the earlier call site silently changes "
            "meaning:\n  " + "\n  ".join(collisions),
        )


if __name__ == "__main__":
    unittest.main()
