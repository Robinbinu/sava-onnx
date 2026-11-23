"""Speech and turns of a WAV file.  python examples/offline.py speech.wav"""
import sys

import sava_onnx as sava

audio = sava.load_wav(sys.argv[1])
model = sava.Sava.from_pretrained(variant="int8")
curves = model.run(audio)
print(f"{(curves.speech >= 0.5).mean():.0%} of frames are speech")
for turn in sava.segment(curves, sava.Policy()):
    if turn.kept:
        print(f"turn {turn.start_seconds:6.2f}s - {turn.end_seconds:6.2f}s  ({turn.reason})")
