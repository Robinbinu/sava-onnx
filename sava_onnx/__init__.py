"""SAVA voice activity and end-of-turn detection on ONNX Runtime.

No PyTorch: numpy and onnxruntime only. Mirrors the API of Inworld's
`inworld-sava` package (run / Session / segment), on the ONNX graphs:

    import sava_onnx as sava

    model = sava.Sava.from_pretrained(variant="int8")   # or Sava("sava_int8.onnx")
    curves = model.run(audio)                            # float32, 16 kHz mono
    turns = sava.segment(curves, sava.Policy())

    stream = sava.Session.from_pretrained(variant="int8")
    for piece in microphone():
        frames = stream.push(piece)
    frames = stream.flush()
"""
from __future__ import annotations

import dataclasses
import wave

import numpy as np
import onnxruntime as ort

__version__ = "1.0.0"

SAMPLE_RATE = 16000
FRAME_SAMPLES = 640  # one 40 ms frame
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
LOOKAHEAD_FRAMES = 3  # a speech verdict arrives 120 ms after its audio
TURN_HORIZONS = (0, 5, 10, 20)  # frames; column k of `turns` is P(turn ended by now + horizon)
REPO_ID = "Binu30/sava-onnx"
VARIANTS = ("fp32", "fp16", "int8")

__all__ = ["SAMPLE_RATE", "FRAME_SAMPLES", "FRAME_SECONDS", "LOOKAHEAD_FRAMES", "TURN_HORIZONS",
           "Curves", "Frames", "Policy", "Turn", "TurnSegmenter", "Sava", "Session",
           "segment", "load_wav", "download"]


def _filename(variant: str, streaming: bool) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    base = "sava_streaming" if streaming else "sava"
    return f"{base}.onnx" if variant == "fp32" else f"{base}_{variant}.onnx"


def download(variant: str = "int8", streaming: bool = False, repo_id: str = REPO_ID,
             revision: str | None = None) -> str:
    """Fetch a graph from the Hugging Face Hub (needs `pip install sava-onnx[hub]`)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, _filename(variant, streaming), revision=revision)


def _session(path, threads: int | None, providers) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])


def _mono_float(audio) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"audio must be one-dimensional 16 kHz mono, got shape {audio.shape}")
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


def load_wav(path) -> np.ndarray:
    """16-bit PCM 16 kHz mono WAV -> float32 in [-1, 1]. Resample anything else first,
    e.g. `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`."""
    with wave.open(str(path), "rb") as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM, got {wf.getframerate()} Hz, "
                             f"{wf.getnchannels()} channel(s), {8 * wf.getsampwidth()}-bit")
        return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0


@dataclasses.dataclass
class Curves:
    """Per-frame probabilities of a whole recording; frame t covers samples
    [640 t, 640 (t + 1)). `speech` is the lookahead verdict on frame t,
    `speech_on_time` the verdict without lookahead, `turns [T, K]` P(the turn
    has ended) at every horizon, `turn` its first column, and `flush_turns
    [lag, K]` the end-of-turn outputs of the lookahead steps after the end."""
    speech: np.ndarray
    turn: np.ndarray
    turns: np.ndarray
    speech_on_time: np.ndarray
    flush_turns: np.ndarray


@dataclasses.dataclass
class Frames:
    """One push's outputs: value i of `speech` judges frame `first + i - lag`;
    `turn`/`turns` are about frame `first + i` (on time)."""
    first: int
    speech: np.ndarray
    turn: np.ndarray
    turns: np.ndarray

    def __len__(self) -> int:
        return self.speech.size


class Sava:
    """Offline: a whole recording in, aligned curves out (sava.onnx graphs)."""

    def __init__(self, path, threads: int | None = 1, providers=None):
        self.session = _session(path, threads, providers)

    @classmethod
    def from_pretrained(cls, variant: str = "int8", repo_id: str = REPO_ID, **kwargs) -> "Sava":
        return cls(download(variant, False, repo_id), **kwargs)

    def run(self, audio) -> Curves:
        audio = _mono_float(audio)
        if audio.size == 0:
            empty = np.zeros(0, np.float32)
            none = np.zeros((0, len(TURN_HORIZONS)), np.float32)
            return Curves(empty, empty, none, empty, none)
        speech, on_time, turns, flush = self.session.run(None, {"pcm": audio[None]})
        return Curves(speech=speech, turn=turns[:, 0], turns=turns,
                      speech_on_time=on_time, flush_turns=flush)


class Session:
    """Streaming: samples in pieces of any size, a verdict per whole 40 ms frame
    (sava_streaming.onnx graphs). Not safe for concurrent pushes; one per stream."""

    def __init__(self, path, threads: int | None = 1, providers=None):
        self.session = _session(path, threads, providers)
        self._names = [i.name for i in self.session.get_inputs() if i.name != "pcm"]
        self.reset()

    @classmethod
    def from_pretrained(cls, variant: str = "int8", repo_id: str = REPO_ID, **kwargs) -> "Session":
        return cls(download(variant, True, repo_id), **kwargs)

    def reset(self) -> None:
        """Start a new stream: zero caches, no pending samples."""
        self._state = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape],
                                        np.int64 if "int64" in i.type else np.float32)
                       for i in self.session.get_inputs() if i.name != "pcm"}
        self._pending = np.zeros(0, np.float32)
        self._frames = 0
        self._closed = False

    @property
    def frames(self) -> int:
        return self._frames

    def push(self, samples) -> Frames:
        if self._closed:
            raise RuntimeError("The session is flushed; call reset() for a new stream.")
        self._pending = np.concatenate([self._pending, _mono_float(samples)])
        whole = self._pending.size - self._pending.size % FRAME_SAMPLES
        chunk, self._pending = self._pending[:whole], self._pending[whole:]
        return self._step(chunk)

    def flush(self) -> Frames:
        """End the stream: pad the last partial frame and step `lag` silent frames
        so the last real frames get their verdicts."""
        if self._closed:
            raise RuntimeError("The session is already flushed.")
        tail = (-self._pending.size) % FRAME_SAMPLES if self._pending.size else 0
        chunk = np.concatenate([self._pending, np.zeros(tail + LOOKAHEAD_FRAMES * FRAME_SAMPLES, np.float32)])
        self._pending = np.zeros(0, np.float32)
        self._closed = True
        return self._step(chunk)

    def _step(self, chunk: np.ndarray) -> Frames:
        first = self._frames
        speech, turns = [], []
        for i in range(0, chunk.size, FRAME_SAMPLES):
            out = self.session.run(None, {"pcm": chunk[None, i:i + FRAME_SAMPLES], **self._state})
            speech.append(out[0][0])
            turns.append(out[1][0])
            self._state = dict(zip(self._names, out[2:]))
        self._frames += len(speech)
        turns = np.array(turns, np.float32).reshape(-1, len(TURN_HORIZONS))
        return Frames(first, np.array(speech, np.float32), turns[:, 0], turns)


# The turn policy below is ported from inworld/sava/segmenter.py (Apache 2.0,
# Copyright 2026 Inworld AI), unchanged in behaviour, adapted to this module's
# Curves and Frames.

@dataclasses.dataclass(frozen=True)
class Policy:
    """Speech gates and timing rules used to open and close turns."""
    speech_threshold: float = 0.5
    turn_threshold: float = 0.9
    pre_roll_ms: int = 800
    min_silence_before_check_ms: int = 0
    max_silence_ms: int = 1600
    min_speech_ms: int = 400
    min_confident_speech_ms: int = 200
    confident_speech_threshold: float = 0.9


@dataclasses.dataclass
class Turn:
    """Frames [start, end) of the audio, pre-roll included; `reason` is how it
    closed ("turn", "silence" or "end"); `kept` whether it passed the gate."""
    start: int
    end: int
    speech_ms: int
    reason: str
    kept: bool

    @property
    def start_seconds(self) -> float:
        return self.start * FRAME_SECONDS

    @property
    def end_seconds(self) -> float:
        return self.end * FRAME_SECONDS


def _ms_to_frames(ms: int) -> int:
    return round(ms / (1000 * FRAME_SECONDS))


class TurnSegmenter:
    """Feed `Frames` in order; returns the turns they close."""

    def __init__(self, policy: Policy | None = None, lag: int = LOOKAHEAD_FRAMES) -> None:
        self.policy = policy or Policy()
        self.lag = lag
        self._start: int | None = None
        self._speech_ms = self._confident_ms = self._pause_frames = 0
        self._last_judged = -1

    @property
    def open(self) -> bool:
        return self._start is not None

    def feed(self, frames: Frames) -> list[Turn]:
        closed = []
        for i in range(len(frames)):
            judged = frames.first + i - self.lag
            if judged < 0:
                continue
            turn = self._frame(judged, frames.first + i, float(frames.speech[i]), float(frames.turn[i]))
            if turn is not None:
                closed.append(turn)
        return closed

    def finish(self) -> list[Turn]:
        if self._start is None:
            return []
        return [self._close(self._last_judged + 1, "end")]

    def _frame(self, judged: int, now: int, speech: float, turn: float) -> Turn | None:
        p = self.policy
        frame_ms = round(1000 * FRAME_SECONDS)
        self._last_judged = judged
        is_speech = speech >= p.speech_threshold
        if self._start is None:
            if not is_speech:
                return None
            self._start = max(0, judged - _ms_to_frames(p.pre_roll_ms))
            self._speech_ms = self._confident_ms = self._pause_frames = 0
        if is_speech:
            self._pause_frames = 0
            self._speech_ms += frame_ms
            if speech >= p.confident_speech_threshold:
                self._confident_ms += frame_ms
            return None
        self._pause_frames += 1
        pause_ms = self._pause_frames * frame_ms
        if pause_ms >= p.max_silence_ms:
            return self._close(now + 1, "silence")
        if pause_ms >= p.min_silence_before_check_ms and turn >= p.turn_threshold:
            return self._close(now + 1, "turn")
        return None

    def _close(self, end: int, reason: str) -> Turn:
        p = self.policy
        kept = self._speech_ms >= p.min_speech_ms or self._confident_ms >= p.min_confident_speech_ms
        turn = Turn(self._start, end, self._speech_ms, reason, kept)
        self._start = None
        return turn


def segment(curves: Curves, policy: Policy | None = None, lag: int = LOOKAHEAD_FRAMES) -> list[Turn]:
    """The turns of a whole recording, as a stream would cut them."""
    seg = TurnSegmenter(policy, lag)
    steps = np.concatenate([curves.turns, curves.flush_turns])[lag:lag + len(curves.speech)]
    frames = Frames(first=lag, speech=np.asarray(curves.speech), turn=steps[:, 0], turns=steps)
    return seg.feed(frames) + seg.finish()
