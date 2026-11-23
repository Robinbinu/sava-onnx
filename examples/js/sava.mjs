// SAVA on onnxruntime-node (the same calls work in onnxruntime-web).
//   npm install && node sava.mjs ../../models/sava_streaming_int8.onnx speech.wav
import fs from "node:fs";
import * as ort from "onnxruntime-node";

const FRAME = 640; // 40 ms at 16 kHz
const LAG = 3; // a speech verdict judges the frame 3 calls ago

// 16-bit PCM, 16 kHz mono WAV -> Float32Array in [-1, 1].
function readWav(path) {
  const b = fs.readFileSync(path);
  let o = 12;
  while (b.toString("ascii", o, o + 4) !== "data") o += 8 + b.readUInt32LE(o + 4);
  const pcm = new Int16Array(b.buffer, b.byteOffset + o + 8, b.readUInt32LE(o + 4) / 2);
  return Float32Array.from(pcm, (s) => s / 32768);
}

export class SavaStream {
  static async create(path) {
    const s = new SavaStream();
    s.session = await ort.InferenceSession.create(path);
    s.reset();
    return s;
  }

  // Zero state: one tensor per non-pcm input, shapes from the graph.
  reset() {
    this.state = {};
    for (const name of this.session.inputNames) {
      if (name === "pcm") continue;
      const meta = this.session.inputMetadata.find((m) => m.name === name);
      const dims = meta.shape.map((d) => (typeof d === "number" ? d : 1));
      const size = dims.reduce((a, d) => a * d, 1);
      this.state[name] = meta.type === "int64"
        ? new ort.Tensor("int64", new BigInt64Array(size), dims)
        : new ort.Tensor("float32", new Float32Array(size), dims);
    }
    this.frames = 0;
  }

  // One 40 ms frame in; P(speech) for frame (this.frames - LAG) and P(turn ended) now.
  async step(frame) {
    const out = await this.session.run({ pcm: new ort.Tensor("float32", frame, [1, FRAME]), ...this.state });
    for (const name of Object.keys(this.state)) this.state[name] = out[`next_${name}`];
    this.frames += 1;
    return { speech: out.speech.data[0], turn: out.turn.data[0] };
  }
}

if (process.argv[1].endsWith("sava.mjs")) {
  const [model, wav] = process.argv.slice(2);
  const audio = readWav(wav);
  const stream = await SavaStream.create(model);
  const padded = new Float32Array(Math.ceil(audio.length / FRAME) * FRAME + LAG * FRAME);
  padded.set(audio);
  let speechFrames = 0;
  const t0 = performance.now();
  for (let i = 0; i < padded.length; i += FRAME) {
    const { speech } = await stream.step(padded.subarray(i, i + FRAME));
    if (stream.frames > LAG && speech >= 0.5) speechFrames += 1;
  }
  const ms = performance.now() - t0;
  const frames = stream.frames - LAG;
  console.log(`${frames} frames, ${((100 * speechFrames) / frames).toFixed(0)}% speech, ` +
              `${(ms / frames).toFixed(2)} ms per 40 ms frame`);
}
