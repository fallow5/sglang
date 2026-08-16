"""Launch paths read the configured parallel sizes, not the live ones.

`get_parallel().pp_size` and its four siblings are read-through properties over
the process groups, so they answer only after distributed init. The launcher
decides how many processes to spawn *before* that, and a live read there raises
`Distributed environment is not initialized` -- a startup crash no unit test
reaches, because nothing short of booting a server runs the launcher.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

# The five sizes ParallelContext shadows with a live property; each has a
# `configured_*` accessor that answers from the config bag instead.
_LIVE_SHADOWED = {
    "tp_size": "configured_tp_size()",
    "pp_size": "configured_pp_size()",
    "moe_dp_size": "configured_moe_dp_size()",
    "attn_cp_size": "configured_attn_cp_size()",
    "dcp_size": "a configured accessor (none exists yet; add one beside configured_pp_size)",
}

# Modules that run before this process joins its process groups.
_PRE_DIST = (
    "srt/entrypoints/engine.py",
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/sidecar.py",
    "srt/ray/engine.py",
    "srt/ray/http_server.py",
    "srt/ray/data_parallel_controller.py",
    "srt/managers/data_parallel_controller.py",
)


def _parallel_bag_names(tree):
    """What this module calls `get_parallel`, plus any runtime_context alias.

    A literal-name match reads only one spelling; an aliased import or a
    module-qualified call is the same read with a different surface.
    """
    names, modules = set(), set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("runtime_context")
        ):
            names |= {
                a.asname or a.name for a in node.names if a.name == "get_parallel"
            }
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("runtime_context"):
                    modules.add(a.asname or a.name.split(".")[0])
    return names, modules


def _is_parallel_bag_call(node, names, modules) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in names
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_parallel"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in modules
    )


class TestLaunchPathsReadConfiguredSizes(CustomTestCase):
    def test_no_live_topology_read_before_distributed_init(self):
        offenders = []
        for rel in _PRE_DIST:
            path = _PACKAGE_ROOT / rel
            tree = ast.parse(path.read_text())
            names, modules = _parallel_bag_names(tree)
            # Local aliases of the bag itself: `p = get_parallel()` then
            # `p.pp_size` is the same read one line later.
            aliases = {
                t.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and _is_parallel_bag_call(n.value, names, modules)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in _LIVE_SHADOWED:
                    base = node.value
                    if _is_parallel_bag_call(base, names, modules) or (
                        isinstance(base, ast.Name) and base.id in aliases
                    ):
                        offenders.append(
                            f"{rel}:{node.lineno} reads the live {node.attr}; "
                            f"use {_LIVE_SHADOWED[node.attr]}"
                        )
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in _LIVE_SHADOWED
                    and (
                        _is_parallel_bag_call(node.args[0], names, modules)
                        or (
                            isinstance(node.args[0], ast.Name)
                            and node.args[0].id in aliases
                        )
                    )
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} reads the live "
                        f"{node.args[1].value} through getattr; "
                        f"use {_LIVE_SHADOWED[node.args[1].value]}"
                    )
        self.assertEqual(
            offenders,
            [],
            "launch paths run before distributed init:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
