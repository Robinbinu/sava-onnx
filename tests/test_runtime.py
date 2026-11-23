"""sava_onnx (numpy + onnxruntime) against inworld-sava (PyTorch): same curves,
same streaming frames, same turns.

    python -m unittest tests/test_runtime.py
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sava_onnx  # noqa: E402
from parity import LINES, tts  # noqa: E402

MODELS = ROOT / "models"


class RuntimeMatchesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inworld import sava
        cls.sava = sava
        cls.ref = sava.load("inworld/sava").eval()
        gap = np.zeros(sava_onnx.SAMPLE_RATE, np.float32)
        cls.audio = np.concatenate([x for v, t in LINES[:3] for x in (tts(v, t), gap)])

    def test_offline_curves(self):
        want = self.sava.run(self.ref, self.audio)
        got = sava_onnx.Sava(MODELS / "sava.onnx").run(self.audio)
        for field in ("speech", "speech_on_time", "turns", "flush_turns"):
            np.testing.assert_allclose(getattr(got, field), getattr(want, field), atol=1e-4, err_msg=field)

    def test_streaming_in_odd_pieces(self):
        want = self.sava.Session(self.ref)
        got = sava_onnx.Session(MODELS / "sava_streaming.onnx")
        rng = np.random.default_rng(1)
        cuts = np.sort(rng.integers(0, self.audio.size, 25))
        ws, gs = [], []
        for piece in np.split(self.audio, cuts):
            ws.append(want.push(piece).speech)
            gs.append(got.push(piece).speech)
        ws.append(want.flush().speech)
        gs.append(got.flush().speech)
        np.testing.assert_allclose(np.concatenate(gs), np.concatenate(ws), atol=1e-4)

    def test_turns_match_the_reference_segmenter(self):
        want = self.sava.segment(self.sava.run(self.ref, self.audio), self.sava.Policy(),
                                 self.ref.lookahead_frames)
        for name in ("sava.onnx", "sava_fp16.onnx", "sava_int8.onnx"):
            with self.subTest(name=name):
                got = sava_onnx.segment(sava_onnx.Sava(MODELS / name).run(self.audio))
                self.assertEqual([(t.reason, t.kept) for t in got], [(t.reason, t.kept) for t in want])
                # int8 is an approximation: a turn may open or close one 40 ms frame apart.
                slack = 1 if "int8" in name else 0
                for g, w in zip(got, want):
                    self.assertLessEqual(abs(g.start - w.start), slack)
                    self.assertLessEqual(abs(g.end - w.end), slack)

    def test_reset_starts_a_fresh_stream(self):
        s = sava_onnx.Session(MODELS / "sava_streaming_int8.onnx")
        a = np.concatenate([s.push(self.audio[:32000]).speech, s.flush().speech])
        s.reset()
        b = np.concatenate([s.push(self.audio[:32000]).speech, s.flush().speech])
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
