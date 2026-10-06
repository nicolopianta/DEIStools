import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deistools.processing.data_loader import find_technique_folders, load_experiment


def _technique(folder, loop, seq, n_blocks, fill, suffix=""):
    d = folder / "pico_aquisition" / f"cycle_{loop}_sequence_{seq}{suffix}"
    d.mkdir(parents=True)
    np.save(d / "impedance.npy", np.full((3, n_blocks), fill + 1j, dtype=np.complex64))
    np.save(d / "voltage.npy", np.full(5 * n_blocks, fill, dtype=np.float32))
    np.save(d / "current.npy", np.full(5 * n_blocks, -fill, dtype=np.float32))
    return d


class LoadExperiment(unittest.TestCase):
    def test_new_layout_concatenates_in_run_order_and_ignores_decoys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata_deis_exp.json").write_text(json.dumps({"Frequencies multisine (Hz)": [1.0, 3.0, 7.0]}))
            (root / "measurement_data.txt").write_text(
                "Time/s\tEwe/V\tI/A\tTechnique_num\tLoop_num\n0\t1.0\t0.0\t0\t0\n1\t1.1\t1e-4\t0\t0\n")
            _technique(root, 0, 2, 3, 3.0)
            _technique(root, 0, 0, 2, 1.0)
            _technique(root, 1, 0, 1, 4.0)
            _technique(root, 0, 1, 0, 2.0)                 # an OCV step with no blocks
            _technique(root, 0, 2, 3, 99.0, suffix="_manual_stop")   # leftover copy from an older version
            data = load_experiment(root)
            self.assertEqual([d.name for d in data.technique_folders],
                             ["cycle_0_sequence_0", "cycle_0_sequence_1", "cycle_0_sequence_2", "cycle_1_sequence_0"])
            self.assertEqual(data.impedance.shape, (3, 2 + 0 + 3 + 1))
            np.testing.assert_array_equal(data.impedance[0].real, [1, 1, 3, 3, 3, 4])
            self.assertEqual(data.decimated_voltage.size, 5 * 6)
            np.testing.assert_array_equal(data.decimated_current[:10], -np.ones(10))
            np.testing.assert_array_equal(data.frequencies, [1.0, 3.0, 7.0])
            np.testing.assert_allclose(data.potential, [1.0, 1.1])
            self.assertEqual(len(data.time), 2)

    def test_run_order_is_by_loop_then_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for loop, seq in ((1, 0), (0, 10), (0, 2), (0, 0)):
                _technique(root, loop, seq, 1, 1.0)
            self.assertEqual([d.name for d in find_technique_folders(root)],
                             ["cycle_0_sequence_0", "cycle_0_sequence_2", "cycle_0_sequence_10", "cycle_1_sequence_0"])

    def test_legacy_layout_and_missing_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "impedance.npy", np.ones((2, 4), dtype=complex))
            np.save(root / "decimated_voltage.npy", np.ones(10, dtype=np.float32))
            np.save(root / "decimated_current.npy", np.ones(10, dtype=np.float32))
            data = load_experiment(root)
            self.assertEqual(data.technique_folders, [])
            self.assertEqual(data.impedance.shape, (2, 4))
            self.assertEqual(data.decimated_voltage.size, 10)
            self.assertIsNone(data.time)
            self.assertIsNone(data.frequencies)
            self.assertTrue(any("not found" in n for n in data.notes))


if __name__ == "__main__":
    unittest.main()
