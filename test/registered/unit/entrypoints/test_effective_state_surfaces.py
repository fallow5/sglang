"""Every serving surface can answer what is running, not only what was asked.

`/server_info` and its gRPC and in-process twins report the startup record. The
facts that change after launch -- the model a weight update swapped in, the
parsers resolved from the chat template -- are reported by the model-info
surface instead, and there is one per entry point: HTTP, gRPC and `Engine`.
Adding a field to one and forgetting the others leaves that entry point's
users with no way to see it, which no test notices because each surface passes
its own tests.

The fields are derived from the control-plane writers rather than listed here:
whatever a process writes with `override` after publication is exactly what can
differ from the record, and a hand-kept list omits the field nobody remembered
to migrate. Each surface must both carry the key and take its value from the
effective config -- reading `server_args.<field>` under the right key reports
the startup value with a straight face.
"""

import ast
import inspect
import pathlib
import unittest

import sglang
from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

# The tokenizer process is the one whose control-plane writes a served request
# can observe; a field it overrides there is a post-launch fact.
_TOKENIZER_WRITERS = (
    "srt/managers/tokenizer_manager.py",
    "srt/managers/tokenizer_control_mixin.py",
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/engine.py",
)

# Written post-publish but reported by their own endpoint or owned elsewhere:
# the HiCache mirror has `GET /hicache/storage-backend`, and the model path and
# served name stay manager attributes that every surface already reports.
_REPORTED_ELSEWHERE = {
    "hicache_storage_backend",
    "hicache_storage_backend_extra_config",
    "model_path",
    "served_model_name",
}


def _overridden_fields() -> set:
    """Fields the tokenizer process writes after publication."""
    fields = set()
    for rel in _TOKENIZER_WRITERS:
        tree = ast.parse((_PACKAGE_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in ("override", "record_config_updates"):
                fields |= {kw.arg for kw in node.keywords if kw.arg}
    return fields - _REPORTED_ELSEWHERE


def _effective_reads_in(source: str, func_name: str) -> set:
    """Fields the function reports *and* reads through the effective config.

    A key whose value comes off the `ServerArgs` record does not count: that is
    the startup value under a name that promises the running one.
    """
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != func_name:
            continue
        reported = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                for inner in ast.walk(value):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in ("config_value", "config_leaf")
                        and inner.args
                        and isinstance(inner.args[0], ast.Constant)
                        and inner.args[0].value == key.value
                    ):
                        reported.add(key.value)
        return reported
    return set()


class TestEffectiveStateSurfaces(CustomTestCase):
    def test_each_entry_point_reports_the_post_launch_facts(self):
        surfaces = {
            "http /model_info": _effective_reads_in(
                (_PACKAGE_ROOT / "srt/entrypoints/http_server.py").read_text(),
                "model_info",
            ),
            "grpc get_model_info": _effective_reads_in(
                (_PACKAGE_ROOT / "srt/entrypoints/grpc_bridge.py").read_text(),
                "get_model_info",
            ),
            "Engine.get_model_info": _effective_reads_in(
                inspect.getsource(Engine.get_model_info).lstrip(), "get_model_info"
            ),
        }
        # What any surface reports effectively, all of them owe their users;
        # what the control plane overrides, every surface owes regardless.
        required = set().union(*surfaces.values()) | _overridden_fields()
        self.assertIn(
            "load_format",
            required,
            "the derivation stopped finding the control-plane writers",
        )
        missing = {
            name: sorted(required - reported)
            for name, reported in surfaces.items()
            if required - reported
        }
        self.assertEqual(
            missing,
            {},
            "a serving surface cannot report what it is running: " f"{missing}",
        )


if __name__ == "__main__":
    unittest.main()
