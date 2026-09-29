"""Reject experimentally mismatched method-comparison panels."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import render_talk_rollout as render


class ComparisonChecks(unittest.TestCase):
    def pair(self):
        settings = dict(method="variational", object="graspit_box", seed=42,
                        friction=.25, beta=.9, max_steps=40,
                        runner="release-comparison", contact_friction="matched",
                        stress_timing="physical", restore_table_contacts=True)
        a = SimpleNamespace(meta={"settings": settings, "release_commit": "same",
                                  "model_sha256": "same"},
                            states={"qpos": np.zeros((2, 3)), "qvel": np.zeros((2, 3))})
        b = copy.deepcopy(a)
        b.meta["settings"]["method"] = "cem"
        return a, b

    def test_matched_methods_accepted(self):
        self.assertTrue(hasattr(render, "validate_comparison"), "Comparison validation is missing")
        render.validate_comparison(self.pair())

    def test_rejects_wrong_comparator_and_changed_conditions(self):
        self.assertTrue(hasattr(render, "validate_comparison"), "Comparison validation is missing")
        for key, value in [("method", "variational"), ("object", "cube"),
                           ("friction", .7), ("seed", 99)]:
            with self.subTest(key=key):
                a, b = self.pair()
                b.meta["settings"][key] = value
                with self.assertRaises(ValueError):
                    render.validate_comparison((a, b))
        a, b = self.pair()
        b.states["qpos"][0, 0] = .01
        with self.assertRaises(ValueError):
            render.validate_comparison((a, b))

    def test_recovered_source_hash_must_match(self):
        a, b = self.pair()
        for r in (a, b):
            del r.meta["release_commit"]
            r.meta["source_sha256"] = "recovered-runner"
        render.validate_comparison((a, b))
        b.meta["source_sha256"] = "different-runner"
        with self.assertRaises(ValueError):
            render.validate_comparison((a, b))


if __name__ == "__main__":
    unittest.main()
