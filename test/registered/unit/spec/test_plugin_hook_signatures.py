"""A plugin hook and the built-in it stands in for take the same arguments.

`CustomSpecAlgo` is the extension point: a registered algorithm's method is
called through the same dispatch as the built-in ones. Nothing in the tree
implements it, so changing an argument list on the built-in side leaves the
hook stale and the first plugin to run hits a TypeError -- which is exactly
what happened when the disaggregation draft-input builder dropped its config
parameter.
"""

import inspect
import unittest

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Methods a registered algorithm may override, which the dispatch calls on
# either type without knowing which it holds.
_DISPATCHED = (
    "build_disagg_draft_input",
    "get_num_tokens_per_req_for_target_verify",
)


class TestPluginHookSignatures(CustomTestCase):
    def test_the_hook_and_the_dispatch_agree(self):
        mismatches = []
        for name in _DISPATCHED:
            hook = getattr(CustomSpecAlgo, name, None)
            builtin = getattr(SpeculativeAlgorithm, name, None)
            if hook is None or builtin is None:
                mismatches.append(f"{name}: missing on one side")
                continue
            hook_params = list(inspect.signature(hook).parameters)
            builtin_params = list(inspect.signature(builtin).parameters)
            if hook_params != builtin_params:
                mismatches.append(
                    f"{name}: CustomSpecAlgo{tuple(hook_params)} vs "
                    f"SpeculativeAlgorithm{tuple(builtin_params)}"
                )
        self.assertEqual(
            mismatches,
            [],
            "a plugin implementing the hook would be called with the "
            "dispatch's arguments:\n  " + "\n  ".join(mismatches),
        )


if __name__ == "__main__":
    unittest.main()
