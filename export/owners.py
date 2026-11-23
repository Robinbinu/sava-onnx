"""Map each weight MatMul/Gemm/Conv in an exported graph to the PyTorch
parameter it came from, by content, since the streaming graph carries no
module scope names (step() calls submodules directly, outside __call__)."""
import numpy as np
import onnx
from onnx import numpy_helper


def _key(a):
    a = np.asarray(a, dtype=np.float64)
    return a.shape, round(float(a.sum()), 3), round(float(np.abs(a).sum()), 3)


def weight_owners(onnx_path, torch_model):
    by_key = {}
    for name, p in torch_model.named_parameters():
        w = p.detach().cpu().numpy()
        by_key[_key(w)] = name
        if w.ndim == 2:
            by_key[_key(w.T)] = name
    for name, b in torch_model.named_buffers():
        w = b.detach().cpu().numpy()
        by_key.setdefault(_key(w), name)
        if w.ndim == 2:
            by_key.setdefault(_key(w.T), name)
    m = onnx.load(str(onnx_path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t)
              for n in m.graph.node if n.op_type == "Constant" and n.attribute and n.attribute[0].t.dims}
    inits.update(consts)
    owners = {}
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm", "Conv"):
            for inp in n.input[1:2]:
                if inp in inits:
                    owners[n.name] = by_key.get(_key(inits[inp]), "?")
    return owners


def fp16_weights(src, dst, owners, keep):
    """Store the weight of every MatMul/Gemm whose owner `keep` rejects as fp16,
    with a Cast back to fp32 in front of it: half the bytes, fp32 arithmetic.

    onnxruntime.transformers.float16 was tried first and is unusable here: with
    every node in its block list it still changed the graph's output by 1.0.
    """
    from onnx import helper
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    casts, done = [], {}
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm") or n.name not in owners or keep(owners[n.name]):
            continue
        w = n.input[1]
        if w not in done:
            if w in inits:
                arr = numpy_helper.to_array(inits[w])
                g.initializer.remove(inits[w])
            elif w in consts:
                arr = numpy_helper.to_array(consts[w].attribute[0].t)
                g.node.remove(consts[w])
            else:
                continue
            half = f"{w}_fp16"
            g.initializer.append(numpy_helper.from_array(arr.astype(np.float16), half))
            casts.append(helper.make_node("Cast", [half], [w], to=onnx.TensorProto.FLOAT, name=f"{w}_cast"))
            done[w] = True
    for c in reversed(casts):
        g.node.insert(0, c)
    onnx.save(m, str(dst))
