"""A process entry publishes before it reads a config namespace.

The functions checked here are found by walking the package for `publish`
calls, not by listing them: a hand-kept list can name a function that no longer
exists and still pass, which is how the Ray actor entry went unchecked.

Every such function starts a process (or is the first thing a spawned worker
runs), so the runtime context it inherits is empty. A bag read placed above the
publish raises `config namespace ... not published` -- in a spawned worker,
which no unit test starts, so the failure only shows up as a server that never
comes up.

The check is textual on purpose: it compares the line of the first bag read in
the function body against the line of the publish, which is the ordering a
reader of that function checks by eye.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

_ACCESSORS = frozenset(
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

# The process entries that must be found by the walk. A derivation that stops
# matching -- an import rewritten, a publish moved behind a helper -- would
# otherwise leave this test green over an empty set.
_KNOWN_ENTRIES = frozenset(
    {
        ("srt/managers/scheduler.py", "run_scheduler_process"),
        ("srt/managers/detokenizer_manager.py", "run_detokenizer_process"),
        (
            "srt/managers/data_parallel_controller.py",
            "run_data_parallel_controller_process",
        ),
        ("srt/ray/scheduler_actor.py", "__init__"),
        ("srt/disaggregation/encode_server.py", "__init__"),
        ("srt/disaggregation/encode_server.py", "launch_server"),
        ("srt/managers/tokenizer_manager.py", "__init__"),
        ("srt/entrypoints/engine.py", "_launch_subprocesses"),
        (
            "srt/elastic_ep/expert_backup_manager.py",
            "run_expert_backup_manager_process",
        ),
        ("srt/weight_cache/daemon.py", "load"),
    }
)

# `publish` itself and its named wrappers live here; a call inside them is the
# definition, not a process entry.
_PUBLISH_HOMES = frozenset({"srt/runtime_context.py", "srt/server_args.py"})

_CONFIG_MODULES = frozenset({"sglang.srt.runtime_context", "sglang.srt.server_args"})


def _bindings(tree):
    """What this module calls the publisher and the bag accessors.

    Resolved from the imports rather than matched by name: a model's
    ``index_topk_share.publish()`` and a platform's ``get_device()`` are
    unrelated methods that a name-only match reports as config calls.
    """
    publishers, accessors, modules = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _CONFIG_MODULES:
            for a in node.names:
                local = a.asname or a.name
                if a.name == "publish" or a.name.startswith("set_global_server_args"):
                    publishers.add(local)
                elif a.name in _ACCESSORS:
                    accessors.add(local)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in _CONFIG_MODULES:
                    modules.add(a.asname or a.name.split(".")[0])
    return publishers, accessors, modules


def _call_name(node, bare: set, modules: set):
    """The called name, if this call goes through one of `bare` / `modules`."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name) and node.func.id in bare:
        return node.func.id
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in modules
    ):
        return node.func.attr
    return None


def _first_line(fn, predicate):
    # ast.walk yields breadth-first, so the first match is not the earliest
    # line; take the minimum.
    lines = [n.lineno for n in ast.walk(fn) if predicate(n)]
    return min(lines) if lines else None


def _publishing_functions():
    """(relative path, function node, is_publish, is_bag_read) per publisher."""
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel in _PUBLISH_HOMES:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        publishers, accessors, modules = _bindings(tree)
        if not publishers:
            continue
        is_publish = lambda n: _call_name(n, publishers, modules) is not None
        is_read = lambda n: _call_name(n, accessors, modules) in accessors
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _first_line(fn, is_publish) is not None:
                yield rel, fn, is_publish, is_read


class TestPublishPrecedesBagReads(CustomTestCase):
    def test_every_publishing_entry_publishes_first(self):
        offenders = []
        found = set()
        for rel, fn, is_publish, is_read in _publishing_functions():
            found.add((rel, fn.name))
            publish_line = _first_line(fn, is_publish)
            read_line = _first_line(fn, is_read)
            if read_line is not None and read_line < publish_line:
                offenders.append(
                    f"{rel}:{fn.name} reads a config namespace at line "
                    f"{read_line}, before its publish at {publish_line}"
                )
        self.assertEqual(
            sorted(_KNOWN_ENTRIES - found),
            [],
            "the walk stopped finding known process entries; the derivation "
            "is broken, not the tree",
        )
        self.assertEqual(
            offenders,
            [],
            "a spawned worker starts with an empty context:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
