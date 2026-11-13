"""Export CNN fixtures with independent batch-one ORT references."""

import argparse
import hashlib
import json
import sys
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


def train_mnist(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1":
        raise ValueError("MNIST checksum mismatch")
    with np.load(path) as data:
        train = torch.from_numpy(data["x_train"][:, None].astype(np.float32) / 255)
        labels = torch.from_numpy(data["y_train"].astype(np.int64))
        test = data["x_test"][:, None].astype(np.float32) / 255
        test_labels = data["y_test"].copy()
    net = nn.Sequential(nn.Conv2d(1, 4, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
                        nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                        nn.Flatten(), nn.Linear(392, 10))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        for indices in torch.randperm(len(train)).split(128):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(net(train[indices]), labels[indices])
            loss.backward()
            optimizer.step()
        print("epoch", epoch + 1, "loss", float(loss.detach()), flush=True)
    rng = np.random.default_rng(81)
    calibration_indices = rng.choice(len(train), 512, replace=False)
    indices = rng.choice(len(test), 256, replace=False)
    arrays = {"x": test[indices], "labels": test_labels[indices], "test_indices": indices,
              "calibration": train[calibration_indices].numpy(),
              "calibration_indices": calibration_indices}
    record = {"training": "MNIST", "epochs": 10, "batch_size": 128,
              "optimizer": "Adam", "learning_rate": 0.001, "device": "cpu",
              "dataset_sha256": digest, "preprocessing": "NCHW float32 pixels / 255",
              "dataset_url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
              "dataset_authors": ["Yann LeCun", "Corinna Cortes"],
              "dataset_license": "CC-BY-SA-3.0",
              "dataset_documentation": "https://keras.io/api/datasets/mnist/",
              "training_samples": 60000, "accuracy_samples": 10000,
              "minimum_float_accuracy": 0.95, "minimum_quantized_accuracy": 0.945,
              "maximum_accuracy_drop": 0.005}
    return net.eval(), arrays, record, test, test_labels


def export(name, out, dataset=None):
    seed = 81
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if name == "mnist":
        if dataset is None:
            raise ValueError("mnist requires --mnist-data")
        net, arrays, training, test, labels = train_mnist(dataset)
    else:
        net = network(name)
        samples = np.random.default_rng(seed).uniform(-1, 1, (768, 4, 8, 8)).astype(np.float32)
        arrays = {"x": samples[512:], "calibration": samples[:512]}
        training = {"training": "none"}
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name + ".onnx")
    torch.onnx.export(net, torch.from_numpy(arrays["x"][:1]), str(path),
                      input_names=["x"], output_names=["y"], dynamo=False, opset_version=20)
    model = onnx.load(path)
    onnx.helper.set_model_props(model, {
        "exporter": "torch.onnx legacy", "torch_version": torch.__version__,
        "seed": str(seed), "training": training["training"], "purpose": "CNN fixture",
        "input_layout": "NCHW", "generator": "tests/fixtures/make_cnn.py",
    })
    onnx.checker.check_model(model)
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = arrays["x"]
    y = np.concatenate([session.run(["y"], {"x": row[None]})[0] for row in x])
    with torch.no_grad():
        np.testing.assert_allclose(net(torch.from_numpy(x)).numpy(), y,
                                   atol=1e-5 if name == "mnist" else 1e-6, rtol=1e-5)
    fixture = path.with_suffix(".npz")
    np.savez(fixture, y=y, **arrays)
    record = {
        "model": name, **training, "seed": seed, "opset": 20,
        "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__, "reference": "ONNX Runtime CPU",
        "calibration_samples": 512, "evaluation_samples": 256,
        "maximum_normalized_mae": 0.025,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, fixture)},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if name == "mnist":
        reference = np.concatenate([session.run(["y"], {"x": row[None]})[0] for row in test])
        with torch.no_grad():
            predicted = torch.cat([net(batch) for batch in torch.from_numpy(test).split(128)]).numpy()
        np.testing.assert_allclose(predicted, reference, atol=1e-5, rtol=1e-4)
        record["float_accuracy"] = float(np.mean(reference.argmax(1) == labels))
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "compiler"))
        import onnx2fpga as otf
        acc = otf.load(path)
        acc.compile(device="k26", mode="stream", hdl="sv", calibration=arrays["calibration"],
                    out_dir=root / "build/mnist_accuracy")
        quantized = acc.predict(test)
        record["quantized_accuracy"] = float(np.mean(quantized.argmax(1) == labels))
        record["normalized_mae"] = float(np.abs(acc.predict(x) - y).mean() / y.std())
        record["weight_bits"], record["activation_bits"] = 8, 8
        record["compiler_sha256"] = hashlib.sha256(b"".join(
            str(p.relative_to(root)).encode() + p.read_bytes()
            for p in sorted((root / "compiler/onnx2fpga").rglob("*.py")))).hexdigest()
        record["passed"] = (record["float_accuracy"] >= record["minimum_float_accuracy"]
                            and record["quantized_accuracy"] >= record["minimum_quantized_accuracy"]
                            and record["float_accuracy"] - record["quantized_accuracy"]
                            <= record["maximum_accuracy_drop"]
                            and record["normalized_mae"] < record["maximum_normalized_mae"])
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, sorted({node.op_type for node in model.graph.node}))
    if not record.get("passed", True):
        raise ValueError("MNIST acceptance failed; see " + str(path.with_suffix(".json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=(*NAMES, "mnist"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--mnist-data", type=Path)
    args = parser.parse_args()
    for name in args.models or NAMES:
        export(name, args.out, args.mnist_data)
