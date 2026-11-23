"""Live end-of-turn detection from the microphone.

    pip install "sava-onnx[hub]" sounddevice
    python examples/microphone.py
"""
import queue

import sounddevice as sd

import sava_onnx as sava

stream = sava.Session.from_pretrained(variant="int8")
turns = sava.TurnSegmenter(sava.Policy())
chunks = queue.Queue()

with sd.InputStream(samplerate=sava.SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=sava.FRAME_SAMPLES, callback=lambda d, *_: chunks.put(d[:, 0].copy())):
    print("Speak; Ctrl-C to stop.")
    try:
        while True:
            frames = stream.push(chunks.get())
            for turn in turns.feed(frames):
                if turn.kept:
                    print(f"end of turn at {turn.end_seconds:.2f}s ({turn.reason})")
            if len(frames):
                print(f"\rspeech {frames.speech[-1]:.2f}  turn {frames.turn[-1]:.2f}", end="")
    except KeyboardInterrupt:
        pass
