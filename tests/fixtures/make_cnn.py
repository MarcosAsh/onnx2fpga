"""Export small, untrained CNNs with independent batch-one ORT references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


OUT = Path(__file__).resolve().parent / "cnn"
NAMES = ("maxpool", "average", "global_average", "grouped", "depthwise", "residual")


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x) + x)


def network(name):
    conv = nn.Conv2d(4, 4, 3, padding=1, groups=2 if name == "grouped" else 1)
    if name == "depthwise":
        conv = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1, groups=4),
                             nn.ReLU(), nn.Conv2d(8, 4, 1))
    elif name == "residual":
        conv = Residual()
    pool = nn.MaxPool2d(2)
    features = 4 * 4 * 4
    if name == "average":
        pool = nn.AvgPool2d(2)
    elif name == "global_average":
        pool = nn.AdaptiveAvgPool2d(1)
        features = 4
    return nn.Sequential(conv, nn.ReLU(), pool, nn.Flatten(), nn.Linear(features, 3)).eval()


def export(name, out):
    seed = 81
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = network(name)
    samples = np.random.default_rng(seed).uniform(-1, 1, (768, 4, 8, 8)).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name + ".onnx")
    torch.onnx.export(net, torch.from_numpy(samples[:1]), str(path),
                      input_names=["x"], output_names=["y"], dynamo=False, opset_version=20)
    model = onnx.load(path)
    onnx.helper.set_model_props(model, {
        "exporter": "torch.onnx legacy", "torch_version": torch.__version__,
        "seed": str(seed), "training": "none", "purpose": "numerical fixture",
        "input_layout": "NCHW", "generator": "tests/fixtures/make_cnn.py",
    })
    onnx.checker.check_model(model)
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = samples[512:]
    y = np.concatenate([session.run(["y"], {"x": row[None]})[0] for row in x])
    with torch.no_grad():
        np.testing.assert_allclose(net(torch.from_numpy(x)).numpy(), y, atol=1e-6, rtol=1e-5)
    fixture = path.with_suffix(".npz")
    np.savez(fixture, x=x, y=y, calibration=samples[:512])
    record = {
        "model": name, "training": "none", "seed": seed, "opset": 20,
        "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__, "reference": "ONNX Runtime CPU",
        "calibration_samples": 512, "evaluation_samples": 256,
        "maximum_normalized_mae": 0.025,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, fixture)},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, sorted({node.op_type for node in model.graph.node}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=NAMES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for name in args.models or NAMES:
        export(name, args.out)
