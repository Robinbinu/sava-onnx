"""Check every ONNX graph against the PyTorch reference and write PARITY.md.

The test audio is generated here (macOS `say` speech, noise, silence, tones),
so the report can be reproduced and nothing private is needed:

    python tests/parity.py --models models
"""
import argparse
import pathlib
import platform
import subprocess
import tempfile
import time
import wave

import numpy as np
import onnxruntime as ort

from inworld import sava

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
SR, FRAME = 16000, 640

LINES = [("Samantha", "Draw me a dragon flying over the mountains."),
         ("Daniel", "Okay. Um. Can you make it a bit bigger, please?"),
         ("Aman", "A cat riding a bicycle at the beach."),
         ("Karen", "No, wait. I want the hard one."),
         ("Moira", "Hello? Is anybody there?")]


def tts(voice, text):
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = f"{tmp}/a.aiff", f"{tmp}/a.wav"
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", str(SR), "-ac", "1",
                        "-c:a", "pcm_s16le", wav], check=True)
        with wave.open(wav) as wf:
            return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768


def clips():
    rng = np.random.default_rng(0)
    out = {}
    speech = [tts(v, t) for v, t in LINES]
    for (voice, _), s in zip(LINES, speech):
        out[f"tts_{voice.lower()}"] = s
    out["silence_2s"] = np.zeros(2 * SR, np.float32)
    out["white_noise_3s"] = (0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    t = np.arange(int(2.5 * SR)) / SR
    out["hum_100hz_2.5s"] = (0.1 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    out["speech_in_noise"] = speech[0] + (0.02 * rng.standard_normal(speech[0].size)).astype(np.float32)
    out["odd_length_1234"] = speech[2][:1234]
    gap = np.zeros(int(1.5 * SR), np.float32)
    long = np.concatenate([np.concatenate([s, gap]) for s in speech] * 6)
    out[f"long_{long.size / SR:.0f}s"] = long  # past the 512-frame attention window
    return out


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run_streaming(sess, audio):
    """Frame by frame, then `lag` silent frames, like sava.Session.flush()."""
    feeds = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape],
                              np.int64 if "int64" in i.type else np.float32)
             for i in sess.get_inputs() if i.name != "pcm"}
    pad = (-audio.size) % FRAME
    audio = np.concatenate([audio, np.zeros(pad + 3 * FRAME, np.float32)])
    speech, turn = [], []
    for i in range(0, audio.size, FRAME):
        outs = sess.run(None, {"pcm": audio[None, i:i + FRAME], **feeds})
        speech.append(outs[0][0])
        turn.append(outs[1][0])
        feeds = dict(zip(feeds, outs[2:]))
    return np.array(speech), np.array(turn)


def compare(ref, got):
    diff = float(np.abs(ref - got).max()) if ref.size else 0.0
    agree = float(np.mean((ref >= 0.5) == (got >= 0.5))) if ref.size else 1.0
    return diff, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--report", default=str(ROOT / "PARITY.md"))
    args = ap.parse_args()
    models = pathlib.Path(args.models)
    model = sava.load("inworld/sava").eval()
    data = clips()
    refs = {}
    for name, audio in data.items():
        curves = sava.run(model, audio)
        s = sava.Session(model)
        a, b = s.push(audio), s.flush()
        refs[name] = (curves, np.concatenate([a.speech, b.speech]), np.concatenate([a.turns, b.turns]))

    rows = []
    for variant in ("sava", "sava_fp16", "sava_int8"):
        sess = session(models / f"{variant}.onnx")
        worst = {"speech": [0, 1], "turn": [0, 1]}
        took = 0.0
        for name, audio in data.items():
            curves = refs[name][0]
            t = time.perf_counter()
            speech, _, turn, flush = sess.run(None, {"pcm": audio[None]})
            turn = np.concatenate([turn, flush])
            took += time.perf_counter() - t
            ref_turn = np.concatenate([curves.turns, curves.flush_turns])
            for key, ref, got in (("speech", curves.speech, speech), ("turn", ref_turn, turn)):
                d, a = compare(ref, got)
                worst[key] = [max(worst[key][0], d), min(worst[key][1], a)]
        rows.append((variant, worst, took / sum(x.size for x in data.values()) * SR))
    for variant in ("sava_streaming", "sava_streaming_fp16", "sava_streaming_int8"):
        sess = session(models / f"{variant}.onnx")
        worst = {"speech": [0, 1], "turn": [0, 1]}
        took = 0.0
        for name, audio in data.items():
            _, ref_speech, ref_turns = refs[name]
            t = time.perf_counter()
            speech, turn = run_streaming(sess, audio)
            took += time.perf_counter() - t
            for key, ref, got in (("speech", ref_speech, speech), ("turn", ref_turns, turn)):
                d, a = compare(ref, got)
                worst[key] = [max(worst[key][0], d), min(worst[key][1], a)]
        rows.append((variant, worst, took / sum(x.size for x in data.values()) * SR))

    lines = ["# Parity with the PyTorch reference", "",
             f"`inworld-sava` {sava.__version__}, onnxruntime {ort.__version__}, 1 CPU thread, "
             f"{platform.machine()} ({platform.system()}). Generated by `tests/parity.py`.", "",
             "Clips: " + ", ".join(f"`{k}` ({v.size / SR:.1f} s)" for k, v in data.items()) + ".", "",
             "Offline graphs are compared with `sava.run`, streaming graphs with `sava.Session` "
             "(push, then flush). *Max diff* is the largest absolute difference in probability over "
             "every frame of every clip; *agreement* is the lowest share of frames on which both "
             "sides fall on the same side of 0.5.", "",
             "| Model | Speech max diff | Speech agreement | Turn max diff | Turn agreement | CPU time per second of audio |",
             "|---|---|---|---|---|---|"]
    for variant, w, per_s in rows:
        lines.append(f"| `{variant}.onnx` | {w['speech'][0]:.1e} | {w['speech'][1]:.2%} | "
                     f"{w['turn'][0]:.1e} | {w['turn'][1]:.2%} | {per_s * 1000:.1f} ms |")
    pathlib.Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
