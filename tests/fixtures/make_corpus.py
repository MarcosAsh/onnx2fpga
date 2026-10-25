"""Exports the models in tests/fixtures/corpus, each with its inputs and ONNX Runtime's outputs.

Needs scikit-learn, skl2onnx, xgboost, lightgbm, onnxmltools, onnx and onnxruntime, and torch
for torch_bn_mlp; pass model names to export only those.
"""

import pathlib
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime

OUT = pathlib.Path(__file__).resolve().parent / "corpus"
FEATURES = 8


def market(raw, count=2000, seed=0):
    """Eight features, roughly unit sized, or at the sizes of prices, volumes and ratios."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (count, FEATURES)) * np.linspace(0.5, 3, FEATURES)
    if raw:
        x = x * [2, 400, 0.1, 30, 1, 0.01, 50, 5] + [100, 5000, 0.5, 0, 20, 0, 250, -3]
    return x.astype(np.float32)


def target(x, seed=1):
    rng = np.random.default_rng(seed)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    return z @ rng.normal(0, 1, FEATURES) + 0.3 * np.tanh(z[:, 0] * z[:, 1]) + rng.normal(
        0, 0.1, len(x))


def save(name, model, x):
    """The model, and the first output ONNX Runtime gives on 256 held out inputs."""
    if model.ir_version > 9:
        model.ir_version = 9
    session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = x[-256:]
    feed = {session.get_inputs()[0].name: inputs}
    y = np.asarray(session.run([session.get_outputs()[0].name], feed)[0], dtype=np.float64)
    OUT.mkdir(exist_ok=True)
    onnx.save(model, OUT / ("%s.onnx" % name))
    np.savez(OUT / ("%s.npz" % name), x=inputs, y=y.reshape(len(inputs), -1),
             calibration=x[:512])
    print("%-20s %6d bytes  %s" % (name, (OUT / ("%s.onnx" % name)).stat().st_size,
                                   " ".join(sorted({n.op_type for n in model.graph.node}))))


def sklearn(name, estimator, raw=False, classify=False, classes=2):
    from skl2onnx import to_onnx

    x = market(raw)
    y = target(x)
    if classify:
        y = np.digitize(y, np.quantile(y, np.linspace(0, 1, classes + 1)[1:-1]))
    estimator.fit(x, y)
    save(name, to_onnx(estimator, x[:1]), x)


def linear_regression():
    from sklearn.linear_model import LinearRegression
    sklearn("linear_regression", LinearRegression())


def ridge():
    from sklearn.linear_model import Ridge
    sklearn("ridge", Ridge(alpha=10.0))


def logistic_binary():
    from sklearn.linear_model import LogisticRegression
    sklearn("logistic_binary", LogisticRegression(), classify=True)


def scaler_linear():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sklearn("scaler_linear", make_pipeline(StandardScaler(), LinearRegression()), raw=True)


def scaler_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sklearn("scaler_logistic", make_pipeline(StandardScaler(), LogisticRegression()),
            raw=True, classify=True)


def logistic_multiclass():
    from sklearn.linear_model import LogisticRegression
    sklearn("logistic_multiclass", LogisticRegression(), classify=True, classes=3)


def mlp_regressor():
    from sklearn.neural_network import MLPRegressor
    sklearn("mlp_regressor", MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=0))


def mlp_sigmoid():
    from sklearn.neural_network import MLPRegressor
    sklearn("mlp_sigmoid", MLPRegressor(hidden_layer_sizes=(16,), activation="logistic",
                                        max_iter=500, random_state=0))


def xgb_small():
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from xgboost import XGBRegressor

    x = market(False)
    model = XGBRegressor(n_estimators=20, max_depth=3, random_state=0).fit(x, target(x))
    save("xgb_small", convert_xgboost(model, initial_types=[
        ("x", FloatTensorType([None, FEATURES]))]), x)


def lgbm_small():
    from lightgbm import LGBMRegressor
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    x = market(False)
    model = LGBMRegressor(n_estimators=20, num_leaves=8, random_state=0, verbose=-1)
    model.fit(x, target(x))
    save("lgbm_small", convert_lightgbm(model, initial_types=[
        ("x", FloatTensorType([None, FEATURES]))], zipmap=False), x)


def torch_bn_mlp():
    """BatchNorm on the raw inputs and after the first layer, exported without folding."""
    import torch

    x = market(True)
    y = target(x)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.BatchNorm1d(FEATURES), torch.nn.Linear(FEATURES, 16),
                              torch.nn.BatchNorm1d(16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    inputs, wanted = torch.tensor(x), torch.tensor(y, dtype=torch.float32)[:, None]
    for _ in range(300):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(net(inputs), wanted).backward()
        optimizer.step()
    net.eval()
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "net.onnx"
        torch.onnx.export(net, inputs[:1], str(path), input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {0: "batch"}}, dynamo=False,
                          do_constant_folding=False)
        save("torch_bn_mlp", onnx.load(path), x)


def ort_static(name, quant_format):
    from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                          quantize_static)
    from skl2onnx import to_onnx
    from sklearn.neural_network import MLPRegressor

    x = market(False)
    net = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(x, target(x))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.feeds = iter([{"X": x[k:k + 1]} for k in range(64)])

        def get_next(self):
            return next(self.feeds, None)

    with tempfile.TemporaryDirectory() as folder:
        source, quantized = pathlib.Path(folder) / "f.onnx", pathlib.Path(folder) / "q.onnx"
        onnx.save(to_onnx(net, x[:1]), source)
        quantize_static(str(source), str(quantized), Reader(), quant_format=quant_format,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        save(name, onnx.load(quantized), x)


def ort_qoperator_mlp():
    from onnxruntime.quantization import QuantFormat
    ort_static("ort_qoperator_mlp", QuantFormat.QOperator)


def ort_qdq_mlp():
    from onnxruntime.quantization import QuantFormat
    ort_static("ort_qdq_mlp", QuantFormat.QDQ)


MODELS = [linear_regression, ridge, logistic_binary, scaler_linear, scaler_logistic,
          logistic_multiclass, mlp_regressor, mlp_sigmoid, xgb_small, lgbm_small, torch_bn_mlp,
          ort_qoperator_mlp, ort_qdq_mlp]

if __name__ == "__main__":
    chosen = sys.argv[1:] or [model.__name__ for model in MODELS]
    for model in MODELS:
        if model.__name__ in chosen:
            model()
