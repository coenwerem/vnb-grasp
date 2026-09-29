"""Check the coupling-aware action projection before running free-body physics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class TransferMapping(unittest.TestCase):
    def test_coupled_motion_round_trip(self):
        path = Path(__file__).with_name("bimanual_grip_transfer.py")
        self.assertTrue(path.exists(), "Bimanual action projection is missing")
        spec = importlib.util.spec_from_file_location("transfer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        S = np.zeros((11, 6))
        S[0, 0] = S[1, 1] = 1
        S[2, 1] = 2.22
        for finger in range(4):
            S[3+2*finger, 2+finger] = 1
            S[4+2*finger, 2+finger] = 1.9
        x = np.array([.01, .02, .03, .04, .05, .06])
        np.testing.assert_allclose(mod.project_hand_action(S @ x), x, atol=1e-12)
        with self.assertRaises(ValueError):
            mod.project_hand_action(np.zeros(12))


if __name__ == "__main__":
    unittest.main()
