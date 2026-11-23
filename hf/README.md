---
license: apache-2.0
base_model: inworld/sava
base_model_relation: quantized
library_name: onnx
pipeline_tag: voice-activity-detection
tags:
- onnx
- onnxruntime
- vad
- voice-activity-detection
- turn-detection
- end-of-turn
- semantic-vad
- streaming
- realtime
- voice-agent
- speech
- audio
- int8
- fp16
- cpu
---

# sava-onnx

**Inworld SAVA voice activity detection and end-of-turn detection, as ONNX.** One small model answers both "is someone speaking?" and "have they finished their turn?" every 40 ms, on CPU, with no PyTorch.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Binu30%2Fsava--onnx-yellow)](https://huggingface.co/Binu30/sava-onnx)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/Robinbinu/sava-onnx/blob/main/LICENSE)
[![Original model](https://img.shields.io/badge/original-inworld%2Fsava-black)](https://huggingface.co/inworld/sava)

[SAVA](https://huggingface.co/inworld/sava) (semantic acoustic voice activity) is Inworld AI's open model that combines streaming VAD with end-of-turn prediction. Inworld reports it ahead of Silero VAD on speech detection and on par with pipecat smart-turn on turn detection ([their benchmarks](https://github.com/inworld-ai/sava/blob/main/bench/README.md)). The official package runs on PyTorch. This repo exports the same weights to ONNX, so SAVA runs wherever ONNX Runtime does: Python, Node.js, the browser (onnxruntime-web), C/C++, C#, Java, Rust, Go, Android and iOS.

> Unofficial community conversion. Not made, reviewed or endorsed by Inworld AI. Weights, behaviour and credit are theirs; see [NOTICE](https://github.com/Robinbinu/sava-onnx/blob/main/NOTICE).

## Models

Six graphs: **offline** (a whole recording in one call) or **streaming** (one 40 ms frame per call, state carried by you), each in three precisions. Download them from [Hugging Face](https://huggingface.co/Binu30/sava-onnx) or the [GitHub release](https://github.com/Robinbinu/sava-onnx/releases).

| File | Use | Size | Matches PyTorch |
|---|---|---|---|
| `sava.onnx` | offline, fp32 | 103 MB | within 5e-6 |
| `sava_fp16.onnx` | offline, fp16 weights | 52 MB | within 5e-6 |
| `sava_int8.onnx` | offline, int8 | **33 MB** | 99.8% of speech verdicts |
| `sava_streaming.onnx` | streaming, fp32 | 103 MB | within 5e-6 |
| `sava_streaming_fp16.onnx` | streaming, fp16 weights | 52 MB | within 5e-6 |
| `sava_streaming_int8.onnx` | streaming, int8 | **32 MB** | 99.8% of speech verdicts |

**Which one?** For a realtime voice agent, or audio arriving in pieces, use `sava_streaming_int8.onnx`. For files and batch jobs, use `sava_int8.onnx`. Choose fp16 for exactly the reference model at half the download (SAVA's checkpoint is bf16, so fp16 storage loses nothing measurable), or fp32 for a runtime without fp16 casts. Full numbers are in [PARITY.md](https://github.com/Robinbinu/sava-onnx/blob/main/PARITY.md).

Speed on one CPU thread (Apple M-series, onnxruntime 1.32): offline is about 16 ms per second of audio; streaming int8 is about 1.6 ms per 40 ms frame in Python and 2.8 ms in Node.js. Budget more on small cloud vCPUs.

## Quick start (Python)

```bash
pip install "sava-onnx[hub] @ git+https://github.com/Robinbinu/sava-onnx"
```

```python
import sava_onnx as sava

audio = sava.load_wav("speech.wav")            # float32, 16 kHz mono
model = sava.Sava.from_pretrained(variant="int8")
curves = model.run(audio)                      # one value per 40 ms frame

print(curves.speech)                           # P(speech) per frame
print(curves.turn)                             # P(turn has ended) per frame

for turn in sava.segment(curves, sava.Policy()):
    if turn.kept:
        print(f"{turn.start_seconds:.2f}s - {turn.end_seconds:.2f}s ({turn.reason})")
```

Streaming, in pieces of any size:

```python
stream = sava.Session.from_pretrained(variant="int8")
turns = sava.TurnSegmenter(sava.Policy())
for piece in audio_chunks():                   # float32 16 kHz mono numpy arrays
    frames = stream.push(piece)
    for turn in turns.feed(frames):
        print("end of turn", turn.end_seconds)
frames = stream.flush()                        # end of stream: last verdicts
```

`sava_onnx` needs only `numpy` and `onnxruntime`. Its `run`, `Session`, `Policy` and `segment` mirror the official `inworld-sava` API, and its turns match the official segmenter exactly with fp32 and fp16 (int8: within one frame), tested in [tests/test_runtime.py](https://github.com/Robinbinu/sava-onnx/blob/main/tests/test_runtime.py). A live microphone demo is in [examples/microphone.py](https://github.com/Robinbinu/sava-onnx/blob/main/examples/microphone.py).

## Quick start (Node.js / browser)

```bash
cd examples/js && npm install
node sava.mjs sava_streaming_int8.onnx speech.wav
```

[examples/js/sava.mjs](https://github.com/Robinbinu/sava-onnx/blob/main/examples/js/sava.mjs) is a dependency-free streaming loop on `onnxruntime-node`. The same calls run in the browser with `onnxruntime-web`.

## Using the graphs directly

All audio is **16 kHz mono float32 in [-1, 1]**. One frame is 640 samples (40 ms). Speech verdicts use 3 frames (120 ms) of lookahead.

### Offline (`sava*.onnx`)

| | Name | Shape | Meaning |
|---|---|---|---|
| in | `pcm` | `[1, samples]` | any length; the graph pads the last frame and the lookahead |
| out | `speech` | `[T]` | P(speech) for frame `t` (lookahead verdict) |
| out | `speech_on_time` | `[T]` | P(speech) for frame `t` without lookahead |
| out | `turn` | `[T, 4]` | P(the turn has ended) at frame `t`, horizons 0, 5, 10, 20 frames |
| out | `turn_flush` | `[3, 4]` | `turn` for the 3 lookahead steps after the end (the segmenter reads them) |

`T = ceil(samples / 640)`, and row `t` of every output is about samples `[640t, 640(t+1))`. Attention memory grows with the square of the clip length, so for recordings longer than a few minutes, use the streaming graph.

### Streaming (`sava_streaming*.onnx`)

Call once per 40 ms frame. Feed `pcm [1, 640]` plus every state tensor, and pass each `next_<name>` output back as `<name>` on the next call. Start with every state tensor at zero. To end a stream, feed 3 frames of silence to get the last verdicts.

| Output | Shape | Meaning |
|---|---|---|
| `speech` | `[1]` | P(speech) for the frame fed **3 calls ago** |
| `turn` | `[1, 4]` | P(the turn has ended) at this frame, horizons 0, 5, 10, 20 frames |
| `next_*` | as below | the state for the next call |

| State | Shape | Type |
|---|---|---|
| `frontend` | `[1, 240]` | float32 |
| `mixer_tail` | `[1, 1, 513]` | float32 |
| `trunk{0,1,2}_conv` | `[1, 3, 1280]` | float32 |
| `trunk{0,1,2}_ssm` | `[1, 16, 64, 128]` | float32 |
| `trunk3_keys`, `trunk3_values`, `vad_keys`, `vad_values`, `turn_keys`, `turn_values` | `[1, 512, 2, 64]` | float32 |
| `trunk3_offset`, `vad_offset`, `turn_offset`, `frames` | `[]` | int64 |
| `prev_speech` | `[1]` | float32 |

The state is fixed-size (three Mamba-2 layers and 512-frame sliding-window attention, about 3.2 MB), so a stream can run indefinitely. Run one state set per audio stream.

## How the export works

`export/export.py` rebuilds every file from the official checkpoint:

```bash
pip install -r export/requirements.txt
python export/export.py --out models
python tests/parity.py --models models     # writes PARITY.md
python -m unittest tests/test_runtime.py
```

Two numerically identical rewrites make SAVA traceable, both in the export script only:
- **Mel frontend.** Its strided unfold and matmul is written as a 1-D convolution with the same DFT kernels; ONNX can't export an unfold over a dynamic length.
- **Sliding-window attention.** It runs as one masked pass instead of a Python loop over 512-frame chunks. Tracing froze that loop at the example's length and cut long clips off at 20.48 s.

**int8** is dynamic per-channel quantization of the attention, feed-forward and head matmuls. The mel frontend, the frame mixer and the Mamba projections stay out of it: quantizing everything cost up to 1.5% of frame verdicts, while leaving those three out cost 0.2%. The weights left out are stored as fp16. **fp16** stores every weight matmul in fp16 and computes in fp32. The usual onnxruntime fp16 converter broke this graph even with every node excluded, so the export does the conversion itself (`export/owners.py`).

Parity is measured on synthetic audio (macOS text-to-speech voices, noise, hum, silence, a 122 s clip past the attention window), so anyone can reproduce it.

## Why GGUF isn't here (yet)

GGUF is only a file format: a model also needs a ggml implementation of its architecture to run, and nothing implements SAVA's (a mel frontend, Mamba-2 layers, sliding-window attention and two heads). A GGUF file today would run nowhere, so none is published. If a ggml runtime (whisper.cpp, for example, which runs Silero VAD that way) adds SAVA, a matching GGUF will follow.

## License and credit

Apache License 2.0, the same as SAVA. The model, its weights and its design are by [Inworld AI](https://inworld.ai): see the [original model](https://huggingface.co/inworld/sava), the [code](https://github.com/inworld-ai/sava) and the [announcement](https://inworld.ai/blog/introducing-sava). Inworld's NOTICE, including the third-party datasets the weights were trained on, is carried unchanged in [NOTICE](https://github.com/Robinbinu/sava-onnx/blob/main/NOTICE), followed by what this repo changed.

If you use SAVA in research, cite Inworld's work; see the [original repository](https://github.com/inworld-ai/sava).
