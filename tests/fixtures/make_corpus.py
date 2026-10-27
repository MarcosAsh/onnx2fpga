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


def normalizer_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    sklearn("normalizer_logistic", make_pipeline(Normalizer("l2"), LogisticRegression()),
            classify=True, classes=3)


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


def classes(x, count=3):
    """The target cut into equal thirds, or halves: down, flat and up."""
    y = target(x)
    return np.digitize(y, np.quantile(y, np.linspace(0, 1, count + 1)[1:-1]))


def xgb_multiclass():
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from xgboost import XGBClassifier

    x = market(False)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(x, classes(x))
    save("xgb_multiclass", convert_xgboost(model, initial_types=[
        ("x", FloatTensorType([None, FEATURES]))]), x)


def lgbm_multiclass():
    from lightgbm import LGBMClassifier
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    x = market(False)
    model = LGBMClassifier(n_estimators=20, num_leaves=8, random_state=0, verbose=-1)
    save("lgbm_multiclass", convert_lightgbm(model.fit(x, classes(x)), initial_types=[
        ("x", FloatTensorType([None, FEATURES]))], zipmap=False), x)


def forest(name, count):
    """A random forest, whose every leaf scores every class."""
    from skl2onnx import to_onnx
    from sklearn.ensemble import RandomForestClassifier

    x = market(False)
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
    save(name, to_onnx(model.fit(x, classes(x, count)), x[:1],
                       options={"zipmap": False}), x)


def forest_multiclass():
    forest("forest_multiclass", 3)


def forest_binary():
    forest("forest_binary", 2)


def ensemble_v5(model):
    """A TreeEnsembleRegressor rewritten as ai.onnx.ml 5's TreeEnsemble, splits and leaves apart."""
    from onnx import TensorProto, helper

    old = model.graph.node[0]
    attrs = {a.name: helper.get_attribute_value(a) for a in old.attribute}
    trees, ids = attrs["nodes_treeids"], attrs["nodes_nodeids"]
    leaf = [m == b"LEAF" for m in attrs["nodes_modes"]]
    splits = {key: k for k, key in enumerate(
        (t, n) for t, n, is_leaf in zip(trees, ids, leaf) if not is_leaf)}
    leaves = {key: k for k, key in enumerate(
        (t, n) for t, n, is_leaf in zip(trees, ids, leaf) if is_leaf)}
    weights = [0.0] * len(leaves)
    for t, n, w in zip(attrs["target_treeids"], attrs["target_nodeids"],
                       attrs["target_weights"]):
        weights[leaves[(t, n)]] += w
    node = {key: [] for key in ("featureids", "splits", "modes", "truenodeids", "trueleafs",
                                "falsenodeids", "falseleafs")}
    for k in (k for k in range(len(trees)) if not leaf[k]):
        node["featureids"].append(attrs["nodes_featureids"][k])
        node["splits"].append(attrs["nodes_values"][k])
        node["modes"].append(["BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "BRANCH_GT", "BRANCH_EQ",
                              "BRANCH_NEQ"].index(attrs["nodes_modes"][k].decode()))
        for side in ("true", "false"):
            child = (trees[k], attrs["nodes_%snodeids" % side][k])
            node["%sleafs" % side].append(int(child in leaves))
            node["%snodeids" % side].append(leaves[child] if child in leaves else splits[child])
    roots = sorted({t for t in trees})
    new = helper.make_node(
        "TreeEnsemble", list(old.input), list(old.output), domain="ai.onnx.ml",
        n_targets=1, aggregate_function=1, post_transform=0,
        tree_roots=[splits[(t, 0)] for t in roots],
        nodes_featureids=node["featureids"], nodes_truenodeids=node["truenodeids"],
        nodes_trueleafs=node["trueleafs"], nodes_falsenodeids=node["falsenodeids"],
        nodes_falseleafs=node["falseleafs"], leaf_targetids=[0] * len(weights),
        nodes_splits=helper.make_tensor("s", TensorProto.FLOAT, [len(node["splits"])],
                                        node["splits"]),
        nodes_modes=helper.make_tensor("m", TensorProto.UINT8, [len(node["modes"])],
                                       node["modes"]),
        leaf_weights=helper.make_tensor("w", TensorProto.FLOAT, [len(weights)], weights))
    graph = helper.make_graph([new], model.graph.name, list(model.graph.input),
                              list(model.graph.output))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18),
                                                   helper.make_opsetid("ai.onnx.ml", 5)],
                             ir_version=9)


def xgb_small_v5():
    """xgb_small, its base value folded into the leaves of the first tree."""
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from xgboost import XGBRegressor

    x = market(False)
    model = XGBRegressor(n_estimators=20, max_depth=3, random_state=0).fit(x, target(x))
    onnx_model = convert_xgboost(model, initial_types=[("x", FloatTensorType([None, FEATURES]))])
    node = onnx_model.graph.node[0]
    base = next((a for a in node.attribute if a.name == "base_values"), None)
    if base is not None:
        first = [k for k, t in enumerate(next(a for a in node.attribute
                                              if a.name == "target_treeids").ints) if t == 0]
        weights = next(a for a in node.attribute if a.name == "target_weights")
        for k in first:
            weights.floats[k] += base.floats[0]
        node.attribute.remove(base)
    save("xgb_small_v5", ensemble_v5(onnx_model), x)


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


def ort_static(name, quant_format, source=None):
    """ONNX Runtime's static quantizer on an MLP, by default skl2onnx's export of a regressor."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                          quantize_static)

    x = market(False)
    if source is None:
        from skl2onnx import to_onnx
        from sklearn.neural_network import MLPRegressor

        net = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=0)
        source = to_onnx(net.fit(x, target(x)), x[:1])

    class Reader(CalibrationDataReader):
        def __init__(self):
            name = source.graph.input[0].name
            self.feeds = iter([{name: x[k:k + 1]} for k in range(64)])

        def get_next(self):
            return next(self.feeds, None)

    with tempfile.TemporaryDirectory() as folder:
        floating, quantized = pathlib.Path(folder) / "f.onnx", pathlib.Path(folder) / "q.onnx"
        onnx.save(source, floating)
        quantize_static(str(floating), str(quantized), Reader(), quant_format=quant_format,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        save(name, onnx.load(quantized), x)


def ort_qoperator_mlp():
    from onnxruntime.quantization import QuantFormat
    ort_static("ort_qoperator_mlp", QuantFormat.QOperator)


def ort_qdq_mlp():
    from onnxruntime.quantization import QuantFormat
    ort_static("ort_qdq_mlp", QuantFormat.QDQ)



def ort_qoperator_gemm():
    """A three class sigmoid MLP as Gemms, which ORT quantizes to QGemm and QLinearSoftmax."""
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantFormat
    from sklearn.neural_network import MLPClassifier

    x = market(False)
    y = target(x)
    y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    net = MLPClassifier(hidden_layer_sizes=(16,), activation="logistic", max_iter=500,
                        random_state=0).fit(x, y)
    weights = [numpy_helper.from_array(array.astype(np.float32), name)
               for name, array in (("w1", net.coefs_[0].T), ("b1", net.intercepts_[0]),
                                   ("w2", net.coefs_[1].T), ("b2", net.intercepts_[1]))]
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
         helper.make_node("Sigmoid", ["h"], ["s"]),
         helper.make_node("Gemm", ["s", "w2", "b2"], ["z"], transB=1),
         helper.make_node("Softmax", ["z"], ["p"], axis=-1)],
        "gemm_mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, FEATURES])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, [None, 3])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    ort_static("ort_qoperator_gemm", QuantFormat.QOperator, model)

MODELS = [linear_regression, ridge, logistic_binary, scaler_linear, scaler_logistic,
          logistic_multiclass, mlp_regressor, mlp_sigmoid, xgb_small, lgbm_small, torch_bn_mlp,
          ort_qoperator_mlp, ort_qdq_mlp, ort_qoperator_gemm, xgb_multiclass,
          lgbm_multiclass, forest_multiclass, forest_binary, xgb_small_v5,
          normalizer_logistic]

if __name__ == "__main__":
    chosen = sys.argv[1:] or [model.__name__ for model in MODELS]
    for model in MODELS:
        if model.__name__ in chosen:
            model()
