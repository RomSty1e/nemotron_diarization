import tempfile
import unittest
from pathlib import Path
import numpy as np
from inference.postprocess import probabilities_to_segments
from inference.rttm import write_rttm


class PostprocessTests(unittest.TestCase):
    def test_overlap_and_tail_clip(self):
        p = np.array([[.9, 0], [.9, .9], [0, .9]])
        s = probabilities_to_segments(p, .08, min_speech_duration=0, audio_duration=.20)
        self.assertEqual(len(s), 2)
        self.assertAlmostEqual(s[0].end, .16)
        self.assertAlmostEqual(s[1].start, .08)
        self.assertAlmostEqual(s[1].end, .20)

    def test_minimum_uses_ceiling(self):
        s = probabilities_to_segments([[.9], [0]], .08, min_speech_duration=.1)
        self.assertEqual(s, [])

    def test_gap_uses_floor(self):
        p = np.array([[.9], [0], [0], [.9]])
        s = probabilities_to_segments(p, .08, min_speech_duration=0, max_gap_duration=.12)
        self.assertEqual(len(s), 2)
        s = probabilities_to_segments(p, .08, min_speech_duration=0, max_gap_duration=.16)
        self.assertEqual(len(s), 1)

    def test_silence_and_empty(self):
        for p in (np.zeros((10, 8)), np.zeros((0, 8))):
            self.assertEqual(probabilities_to_segments(p, .08), [])

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            probabilities_to_segments([[float('nan')]], .08)

    def test_rttm(self):
        s = probabilities_to_segments([[.9], [.9]], .08)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'out.rttm'
            write_rttm(s, 'test', p)
            fields = p.read_text().split()
            self.assertEqual(len(fields), 10)
            self.assertEqual(fields[:5], ['SPEAKER', 'test', '1', '0.000', '0.160'])


if __name__ == '__main__':
    unittest.main()
