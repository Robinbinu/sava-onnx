"""Export Inworld SAVA to ONNX: offline and streaming graphs, fp32/fp16/int8.

    pip install -r export/requirements.txt
    python export/export.py --out models

Every graph is checked against the PyTorch reference by tests/parity.py.
"""
import argparse
import dataclasses
import pathlib

import onnx
import torch
from torch.nn import functional

from inworld import sava
from inworld.sava import config, mel, modules

from owners import fp16_weights, weight_owners

FRAME = config.FRAME_SAMPLES  # 640 samples = 40 ms at 16 kHz
OPSET = 17


def _features_conv(self, pcm):
    """MelFrontend.features with the strided unfold + GEMM written as a conv1d.

    Identical arithmetic (each output is one window dotted with one DFT
    column), but Unfold over a dynamic length has no ONNX export and Conv does.
    """
    dft, bank = self._bases_on(pcm.device)
    spec = functional.conv1d(pcm.float().unsqueeze(1), dft.T.unsqueeze(1),
                             stride=config.MEL_HOP).transpose(1, 2)
    real, imag = spec.chunk(2, dim=-1)
    log_mel = mel.log_compress(torch.matmul(real * real + imag * imag, bank))
    batch, hops, _ = log_mel.shape
    return log_mel.reshape(batch, hops // config.MEL_HOPS_PER_FRAME, -1)


def _one_pass(module, x, window):
    """modules.stream_in_windows as a single pass over the whole sequence.

    The reference walks the sequence 512 frames at a time in a Python loop,
    which tracing freezes at the example's length, so a traced graph silently
    stopped at 20.48 s. windowed_attention masks every key outside a query's
    window, so one pass with an empty cache computes the same thing. Its cost
    is attention memory that grows with the square of the clip length; long
    recordings belong on the streaming graph.
    """
    state = module.init_state(x.shape[0], x.device, x.dtype)
    state = dataclasses.replace(state, keys=state.keys[:, :0], values=state.values[:, :0])
    return module.forward_chunk(x, state)[0]


class Offline(torch.nn.Module):
    """Whole recording in, one row per 40 ms frame out, already aligned.

    pcm [1, S] float32 in [-1, 1], any S >= 1. The graph pads the last partial
    frame and the lookahead itself, so frame t of every output is about audio
    samples [640 t, 640 (t + 1)). The outputs mirror sava.Curves:
    speech [T], speech_on_time [T], turn [T, K] and turn_flush [lag, K], the
    end-of-turn outputs of the lookahead steps after the last frame (the
    reference segmenter reads them).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lag = model.lookahead_frames

    def forward(self, pcm):
        samples = pcm.shape[1]
        frames = (samples + FRAME - 1) // FRAME
        pcm = functional.pad(pcm, (0, (frames + self.lag) * FRAME - samples))
        out = self.model(pcm)
        speech = torch.sigmoid(out.lookahead[0].float())[self.lag:self.lag + frames]
        on_time = torch.sigmoid(out.speech[0].float())[:frames]
        turns = torch.sigmoid(out.turn[0].float())
        return speech, on_time, turns[:frames], turns[frames:frames + self.lag]


def _state_tensors(state):
    t = [state.frontend, state.mixer.tail]
    for block in state.trunk:
        t += [block.conv_tail, block.ssm] if hasattr(block, "ssm") else [block.keys, block.values, block.offset]
    t += [state.vad.keys, state.vad.values, state.vad.offset,
          state.turn.keys, state.turn.values, state.turn.offset,
          state.prev_speech, state.frames]
    return t


def _state_names(model):
    names = ["frontend", "mixer_tail"]
    for i, kind in enumerate(model.cfg.layers):
        names += ([f"trunk{i}_conv", f"trunk{i}_ssm"] if kind == "mamba"
                  else [f"trunk{i}_keys", f"trunk{i}_values", f"trunk{i}_offset"])
    return names + ["vad_keys", "vad_values", "vad_offset", "turn_keys", "turn_values",
                    "turn_offset", "prev_speech", "frames"]


def _rebuild(model, flat):
    from inworld.sava import model as model_lib, modules
    it = iter(flat)
    frontend, tail = next(it), next(it)
    trunk = []
    for kind in model.cfg.layers:
        if kind == "mamba":
            trunk.append(modules.MambaState(conv_tail=next(it), ssm=next(it)))
        else:
            trunk.append(modules.AttentionState(keys=next(it), values=next(it), offset=next(it)))
    vad = modules.AttentionState(keys=next(it), values=next(it), offset=next(it))
    turn = modules.AttentionState(keys=next(it), values=next(it), offset=next(it))
    return model_lib.StreamState(frontend=frontend, mixer=modules.MixerState(tail=tail), trunk=trunk,
                                 vad=vad, turn=turn, prev_speech=next(it), frames=next(it))


class Streaming(torch.nn.Module):
    """One 40 ms frame per call, the caches carried in and out.

    Inputs: pcm [1, 640] float32 plus the state tensors (all zeros to start).
    Outputs: speech [1] (judges the frame `lookahead` calls ago), turn [1, K]
    (end of turn by this frame, one column per horizon), then the next state.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pcm, *flat):
        out, state = self.model.step(pcm, _rebuild(self.model, flat))
        speech = torch.sigmoid(out.lookahead[0].float())
        turn = torch.sigmoid(out.turn[0].float())
        return (speech, turn, *_state_tensors(state))


def export_offline(model, path):
    torch.onnx.export(Offline(model), (torch.zeros(1, FRAME * 80 + 123),), str(path),
                      input_names=["pcm"], output_names=["speech", "speech_on_time", "turn", "turn_flush"],
                      dynamic_axes={"pcm": {1: "samples"}, "speech": {0: "frames"},
                                    "speech_on_time": {0: "frames"}, "turn": {0: "frames"}},
                      opset_version=OPSET, dynamo=False)


def export_streaming(model, path):
    state = model.init_state(1, torch.device("cpu"), torch.float32)
    names = _state_names(model)
    torch.onnx.export(Streaming(model), (torch.zeros(1, FRAME), *_state_tensors(state)), str(path),
                      input_names=["pcm", *names],
                      output_names=["speech", "turn", *[f"next_{n}" for n in names]],
                      opset_version=OPSET, dynamo=False)


def _frontend(owner):
    """The mel frontend's DFT and filterbank are computed buffers, not
    parameters, so they are the only weights with no owner."""
    return owner == "?"


def _keeps_fp32(owner):
    """What int8 must not touch: the frontend, the frame mixer and the Mamba
    projections. Quantizing every matmul cost up to 1.5% of frame verdicts on
    the parity clips; keeping these three cost none (export/README.md)."""
    return _frontend(owner) or owner.startswith("mixer.") or ".mixer." in owner


def quantize_int8(src, dst, model):
    """Dynamic int8 (per-channel weights) for the attention, feed-forward and
    head matmuls; the weights int8 leaves alone are then stored as fp16."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = dst.with_suffix(".pre.onnx")
    mid = dst.with_suffix(".mid.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    owners = weight_owners(pre, model)
    quantize_dynamic(str(pre), str(mid), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"],
                     nodes_to_exclude=[n for n, o in owners.items() if _keeps_fp32(o)])
    fp16_weights(mid, dst, weight_owners(mid, model), _frontend)
    pre.unlink()
    mid.unlink()


def to_fp16(src, dst, model):
    """fp16 storage for every weight matmul, fp32 arithmetic. SAVA's checkpoint
    is bf16, so on the parity clips this changes no output at all."""
    fp16_weights(src, dst, weight_owners(src, model), _frontend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--source", default="inworld/sava")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mel.MelFrontend.features = _features_conv
    modules.stream_in_windows = _one_pass
    model = sava.load(args.source).eval()
    with torch.no_grad():
        export_offline(model, out / "sava.onnx")
        export_streaming(model, out / "sava_streaming.onnx")
    for name in ("sava", "sava_streaming"):
        quantize_int8(out / f"{name}.onnx", out / f"{name}_int8.onnx", model)
        to_fp16(out / f"{name}.onnx", out / f"{name}_fp16.onnx", model)
    for p in sorted(out.glob("*.onnx")):
        print(f"{p.name:28} {p.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
