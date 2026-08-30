"""Train a model, compile it, and find out what quantization cost.

Everything else in this repository starts from an .onnx file that already
exists. This starts from nothing, so the whole path is visible in one file:
make data, fit a model to it, export it, compile it, and then ask the only
question that matters at the end, which is whether the integer design still
gets the same answers as the float model it came from.

There is no framework here on purpose. Training is thirty lines of numpy, the
same dependency the compiler has, so this runs anywhere the compiler does.

    python3 examples/worked_example.py
    python3 examples/worked_example.py --device z7020 --target-cycles 32
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "examples")]

from onnx2fpga.compile import Compiler                       # noqa: E402
from onnx2fpga.p1_ingest.onnx_model import ModelBuilder      # noqa: E402
from onnx2fpga.reference.runner import GraphRunner           # noqa: E402

FEATURES = 8
HIDDEN = 16
CLASSES = 3


def dataset(count, seed, separation=4.0, noise=1.0):
    """A separable problem, so the accuracy at the end means something.

    The class centres are placed rather than drawn: three random centres in
    eight dimensions land on top of each other often enough that the model
    plateaus around 60%, and then nobody can tell whether quantization cost
    anything or the model was simply bad. A task the float model solves is the
    only one that answers the question this example exists to ask.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((CLASSES, FEATURES))
    for index in range(CLASSES):
        centres[index, index] = separation
        centres[index, (index + 1) % FEATURES] = -separation / 2
    labels = rng.integers(0, CLASSES, count)
    values = centres[labels] + rng.normal(0, noise, (count, FEATURES))
    return values.astype(np.float64), labels


def train(values, labels, epochs=400, rate=0.05, seed=0):
    """One hidden layer, relu, softmax cross entropy, plain gradient descent."""
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, np.sqrt(2.0 / FEATURES), (FEATURES, HIDDEN))
    b0 = np.zeros(HIDDEN)
    w1 = rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, CLASSES))
    b1 = np.zeros(CLASSES)

    onehot = np.zeros((len(labels), CLASSES))
    onehot[np.arange(len(labels)), labels] = 1.0

    for _ in range(epochs):
        hidden = np.maximum(values @ w0 + b0, 0.0)
        scores = hidden @ w1 + b1
        shifted = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)

        dscores = (probs - onehot) / len(labels)
        dw1 = hidden.T @ dscores
        db1 = dscores.sum(axis=0)
        dhidden = (dscores @ w1.T) * (hidden > 0)
        dw0 = values.T @ dhidden
        db0 = dhidden.sum(axis=0)

        w0 -= rate * dw0
        b0 -= rate * db0
        w1 -= rate * dw1
        b1 -= rate * db1
    return w0, b0, w1, b1


def predict(weights, values):
    w0, b0, w1, b1 = weights
    return np.maximum(values @ w0 + b0, 0.0) @ w1 + b1


def export(weights, path):
    """The same ONNX any exporter would write: two Gemms and a Relu."""
    w0, b0, w1, b1 = weights
    builder = ModelBuilder("worked").add_input("x", (1, FEATURES))
    builder.add_initializer("w0", w0.astype(np.float32))
    builder.add_initializer("b0", b0.astype(np.float32))
    builder.add_initializer("w1", w1.astype(np.float32))
    builder.add_initializer("b1", b1.astype(np.float32))
    builder.add_node("Gemm", ["x", "w0", "b0"], ["h_pre"], name="fc0")
    builder.add_node("Relu", ["h_pre"], ["h"], name="relu0")
    builder.add_node("Gemm", ["h", "w1", "b1"], ["y"], name="fc1")
    model = builder.add_output("y", (1, CLASSES)).build()
    if path:
        model.save(path)
    return model


def integer_predictions(result, values):
    """Run the compiled design's own golden model, which is exactly what the
    RTL does, and bring the answers back to floats the way a host would."""
    graph, plan = result.hardware_graph, result.plan
    source, sink = graph.inputs[0], graph.outputs[0]
    runner = GraphRunner(graph)
    out = []
    for row in values:
        integers = plan.spec(source).quantize(row[None, :], plan.act_dtype)
        produced = runner.run({source: integers})[sink]
        out.append((produced - plan.zero_point(sink)) * plan.scale(sink))
    return np.concatenate(out, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="vu9p")
    parser.add_argument("--target-cycles", type=int, default=64)
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "build" / "worked")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    train_x, train_y = dataset(600, seed=1)
    test_x, test_y = dataset(200, seed=2)

    weights = train(train_x, train_y)
    float_scores = predict(weights, test_x)
    float_accuracy = float(np.mean(float_scores.argmax(axis=1) == test_y))

    args.out.mkdir(parents=True, exist_ok=True)
    model = export(weights, args.out / "worked.onnx")

    # Calibration wants real inputs, and the training set is what there is.
    samples = [{"x": row[None, :]} for row in train_x[:64]]
    result = Compiler(device=args.device, target_cycles=args.target_cycles,
                      verbose=False).compile(model, samples, args.out)

    integer_scores = integer_predictions(result, test_x)
    integer_accuracy = float(np.mean(integer_scores.argmax(axis=1) == test_y))
    agreement = float(np.mean(
        integer_scores.argmax(axis=1) == float_scores.argmax(axis=1)))

    if not args.quiet:
        print("trained a %d-%d-%d classifier on %d samples"
              % (FEATURES, HIDDEN, CLASSES, len(train_x)))
        print("exported  %s" % (args.out / "worked.onnx"))
        print()
        print(result.folding.render())
        if result.accuracy:
            print()
            print(result.accuracy.render())
        print()
        print("accuracy on %d held out samples" % len(test_x))
        print("  float model     %.1f%%" % (100 * float_accuracy))
        print("  compiled design %.1f%%" % (100 * integer_accuracy))
        print("  same prediction on %.1f%% of them" % (100 * agreement))
        print()
        print("the design is in %s; build and simulate it with" % args.out)
        print("  cd %s && make run" % args.out)
    return {"float_accuracy": float_accuracy,
            "integer_accuracy": integer_accuracy,
            "agreement": agreement,
            "result": result}


if __name__ == "__main__":
    main()
