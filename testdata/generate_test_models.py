"""Generate small ONNX test models and expected outputs via onnxruntime."""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

OPSET = 18
BASE = os.path.dirname(os.path.abspath(__file__))


def save_test_case(name, model, inputs, expected_outputs):
    dir_path = os.path.join(BASE, "ops", name)
    os.makedirs(dir_path, exist_ok=True)
    onnx.save(model, os.path.join(dir_path, "model.onnx"))
    for i, (iname, arr) in enumerate(inputs.items()):
        tp = numpy_helper.from_array(arr, name=iname)
        with open(os.path.join(dir_path, f"input_{i}.pb"), "wb") as f:
            f.write(tp.SerializeToString())
    for i, (oname, arr) in enumerate(expected_outputs.items()):
        tp = numpy_helper.from_array(arr, name=oname)
        with open(os.path.join(dir_path, f"output_{i}.pb"), "wb") as f:
            f.write(tp.SerializeToString())


def run_ort(model, inputs):
    sess = ort.InferenceSession(model.SerializeToString())
    names = [o.name for o in sess.get_outputs()]
    results = sess.run(names, inputs)
    return dict(zip(names, results))


def make_model_simple(graph, opset=OPSET):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def make_unary_case(name, op_type, x, attrs=None, opset=OPSET, dtype=TensorProto.FLOAT, out_dtype=None):
    attrs = attrs or {}
    if out_dtype is None:
        out_dtype = dtype
    node = helper.make_node(op_type, ["X"], ["Y"], **attrs)
    graph = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info("X", dtype, list(x.shape))],
        [helper.make_tensor_value_info("Y", out_dtype, None)],
    )
    model = make_model_simple(graph, opset=opset)
    inputs = {"X": x}
    save_test_case(name, model, inputs, run_ort(model, inputs))


def make_binary_case(name, op_type, a, b, attrs=None, opset=OPSET, dtype=TensorProto.FLOAT, out_dtype=None):
    attrs = attrs or {}
    if out_dtype is None:
        out_dtype = dtype
    node = helper.make_node(op_type, ["A", "B"], ["Y"], **attrs)
    graph = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info("A", dtype, list(a.shape)),
         helper.make_tensor_value_info("B", dtype, list(b.shape))],
        [helper.make_tensor_value_info("Y", out_dtype, None)],
    )
    model = make_model_simple(graph, opset=opset)
    inputs = {"A": a, "B": b}
    save_test_case(name, model, inputs, run_ort(model, inputs))


# ── 1. Add with broadcast ──
def gen_add_broadcast():
    np.random.seed(42)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    node = helper.make_node("Add", ["A", "B"], ["Y"])
    graph = helper.make_graph(
        [node],
        "add_broadcast",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
    )
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("add_broadcast", model, inputs, run_ort(model, inputs))


# ── 2. Sub + Div ──
def gen_sub_div():
    np.random.seed(43)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32) + 2.0  # avoid div by zero
    n1 = helper.make_node("Sub", ["A", "B"], ["T"])
    n2 = helper.make_node("Div", ["T", "B"], ["Y"])
    graph = helper.make_graph(
        [n1, n2],
        "sub_div",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
    )
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("sub_div", model, inputs, run_ort(model, inputs))


# ── 3. MatMul 2D ──
def gen_matmul_2d():
    np.random.seed(44)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph(
        [node],
        "matmul_2d",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
    )
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("matmul_2d", model, inputs, run_ort(model, inputs))


# ── 4. Gemm with transB and bias ──
def gen_gemm_bias():
    np.random.seed(45)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)  # transB=1
    c = np.random.randn(4).astype(np.float32)
    node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
                            alpha=1.0, beta=1.0, transA=0, transB=1)
    graph = helper.make_graph(
        [node],
        "gemm_bias",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3]),
         helper.make_tensor_value_info("C", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
    )
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b, "C": c}
    save_test_case("gemm_bias", model, inputs, run_ort(model, inputs))


# ── 5. Relu ──
def gen_relu():
    np.random.seed(46)
    x = np.random.randn(2, 3).astype(np.float32)
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "relu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("relu", model, inputs, run_ort(model, inputs))


# ── 6. Sigmoid → Tanh chain ──
def gen_sigmoid_tanh():
    np.random.seed(47)
    x = np.random.randn(2, 4).astype(np.float32)
    n1 = helper.make_node("Sigmoid", ["X"], ["T"])
    n2 = helper.make_node("Tanh", ["T"], ["Y"])
    graph = helper.make_graph(
        [n1, n2], "sigmoid_tanh",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("sigmoid_tanh", model, inputs, run_ort(model, inputs))


# ── 7. Softmax ──
def gen_softmax():
    np.random.seed(48)
    x = np.random.randn(1, 5).astype(np.float32)
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1)
    graph = helper.make_graph(
        [node], "softmax",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("softmax", model, inputs, run_ort(model, inputs))


# ── 8. Conv2D ──
def gen_conv2d():
    np.random.seed(49)
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    w = np.random.randn(1, 1, 3, 3).astype(np.float32)
    b = np.random.randn(1).astype(np.float32)
    node = helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    graph = helper.make_graph(
        [node], "conv2d",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5]),
         helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 5, 5])],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "W": w, "B": b}
    save_test_case("conv2d", model, inputs, run_ort(model, inputs))


# ── 9. MaxPool ──
def gen_maxpool():
    np.random.seed(50)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    node = helper.make_node("MaxPool", ["X"], ["Y"],
                            kernel_shape=[2, 2], strides=[2, 2])
    graph = helper.make_graph(
        [node], "maxpool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("maxpool", model, inputs, run_ort(model, inputs))


# ── 10. BatchNormalization ──
def gen_batchnorm():
    np.random.seed(51)
    x = np.random.randn(1, 2, 3, 3).astype(np.float32)
    scale = np.abs(np.random.randn(2).astype(np.float32)) + 0.1
    bias = np.random.randn(2).astype(np.float32)
    mean = np.random.randn(2).astype(np.float32)
    var = np.abs(np.random.randn(2).astype(np.float32)) + 0.1
    node = helper.make_node("BatchNormalization",
                            ["X", "scale", "B", "mean", "var"], ["Y"])
    graph = helper.make_graph(
        [node], "batchnorm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 3]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("var", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "scale": scale, "B": bias, "mean": mean, "var": var}
    save_test_case("batchnorm", model, inputs, run_ort(model, inputs))


# ── 11. Reshape ──
def gen_reshape():
    np.random.seed(52)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 12], dtype=np.int64)
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    graph = helper.make_graph(
        [node], "reshape",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "shape": shape}
    save_test_case("reshape", model, inputs, run_ort(model, inputs))


# ── 12. Transpose ──
def gen_transpose():
    np.random.seed(53)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    node = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    graph = helper.make_graph(
        [node], "transpose",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 3])],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("transpose", model, inputs, run_ort(model, inputs))


# ── 13. GlobalAveragePool ──
def gen_globalavgpool():
    np.random.seed(54)
    x = np.random.randn(1, 3, 4, 4).astype(np.float32)
    node = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "globalavgpool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("globalavgpool", model, inputs, run_ort(model, inputs))


# ── 14. Concat ──
def gen_concat():
    np.random.seed(55)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    node = helper.make_node("Concat", ["A", "B"], ["Y"], axis=0)
    graph = helper.make_graph(
        [node], "concat",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])],
    )
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("concat", model, inputs, run_ort(model, inputs))


# ── 15. Gather ──
def gen_gather():
    np.random.seed(56)
    data = np.random.randn(5, 3).astype(np.float32)
    indices = np.array([1, 3], dtype=np.int64)
    node = helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)
    graph = helper.make_graph(
        [node], "gather",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [5, 3]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"data": data, "indices": indices}
    save_test_case("gather", model, inputs, run_ort(model, inputs))


# ── 16. Squeeze + Unsqueeze ──
def gen_squeeze_unsqueeze():
    np.random.seed(57)
    x = np.random.randn(1, 3, 1, 4).astype(np.float32)
    axes_sq = np.array([0, 2], dtype=np.int64)
    axes_usq = np.array([0, 2], dtype=np.int64)
    n1 = helper.make_node("Squeeze", ["X", "axes_sq"], ["T"])
    n2 = helper.make_node("Unsqueeze", ["T", "axes_usq"], ["Y"])
    graph = helper.make_graph(
        [n1, n2], "squeeze_unsqueeze",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 1, 4]),
         helper.make_tensor_value_info("axes_sq", TensorProto.INT64, [2]),
         helper.make_tensor_value_info("axes_usq", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes_sq": axes_sq, "axes_usq": axes_usq}
    save_test_case("squeeze_unsqueeze", model, inputs, run_ort(model, inputs))


# ── 17. Small MLP (Gemm→Relu→Gemm→Softmax) ──
def gen_mlp_small():
    np.random.seed(58)
    W1 = np.random.randn(8, 4).astype(np.float32) * 0.5
    b1 = np.random.randn(8).astype(np.float32) * 0.1
    W2 = np.random.randn(3, 8).astype(np.float32) * 0.5
    b2 = np.random.randn(3).astype(np.float32) * 0.1
    x = np.random.randn(1, 4).astype(np.float32)

    n1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1)
    n2 = helper.make_node("Relu", ["H"], ["H2"])
    n3 = helper.make_node("Gemm", ["H2", "W2", "b2"], ["L"], transB=1)
    n4 = helper.make_node("Softmax", ["L"], ["Y"], axis=-1)

    graph = helper.make_graph(
        [n1, n2, n3, n4], "mlp_small",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
        initializer=[
            numpy_helper.from_array(W1, "W1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(W2, "W2"),
            numpy_helper.from_array(b2, "b2"),
        ],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("mlp_small", model, inputs, run_ort(model, inputs))


# ── 18. Conv→Relu→MaxPool chain ──
def gen_conv_relu_pool():
    np.random.seed(59)
    x = np.random.randn(1, 1, 8, 8).astype(np.float32)
    w = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.5
    b = np.random.randn(2).astype(np.float32) * 0.1

    n1 = helper.make_node("Conv", ["X", "W", "B"], ["C"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n2 = helper.make_node("Relu", ["C"], ["R"])
    n3 = helper.make_node("MaxPool", ["R"], ["Y"],
                          kernel_shape=[2, 2], strides=[2, 2])

    graph = helper.make_graph(
        [n1, n2, n3], "conv_relu_pool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(b, "B"),
        ],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("conv_relu_pool", model, inputs, run_ort(model, inputs))


def gen_identity():
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    make_unary_case("identity", "Identity", x)


def gen_leakyrelu():
    x = np.array([[-2.0, -0.5, 1.0]], dtype=np.float32)
    make_unary_case("leakyrelu", "LeakyRelu", x, attrs={"alpha": 0.1})


def gen_hardsigmoid():
    x = np.array([[-4.0, 0.0, 4.0]], dtype=np.float32)
    make_unary_case("hardsigmoid", "HardSigmoid", x, attrs={"alpha": 0.2, "beta": 0.5})


def gen_hardswish():
    x = np.array([[-4.0, 0.0, 4.0]], dtype=np.float32)
    make_unary_case("hardswish", "HardSwish", x, opset=14)


def gen_elu():
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    make_unary_case("elu", "Elu", x, attrs={"alpha": 1.0})


def gen_log():
    x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    make_unary_case("log", "Log", x)


def gen_abs():
    x = np.array([[-1.0, 2.0, -3.0]], dtype=np.float32)
    make_unary_case("abs", "Abs", x)


def gen_neg():
    x = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    make_unary_case("neg", "Neg", x)


def gen_sqrt():
    x = np.array([[1.0, 4.0, 9.0]], dtype=np.float32)
    make_unary_case("sqrt", "Sqrt", x)


def gen_exp():
    x = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    make_unary_case("exp", "Exp", x)


def gen_floor():
    x = np.array([[1.2, -0.4, 3.9]], dtype=np.float32)
    make_unary_case("floor", "Floor", x)


def gen_sin():
    x = np.array([[0.0, np.pi / 2, np.pi]], dtype=np.float32)
    make_unary_case("sin", "Sin", x)


def gen_cos():
    x = np.array([[0.0, np.pi / 2, np.pi]], dtype=np.float32)
    make_unary_case("cos", "Cos", x)


def gen_reciprocal():
    x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    make_unary_case("reciprocal", "Reciprocal", x)


def gen_pow():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0, 2.0]], dtype=np.float32)
    make_binary_case("pow", "Pow", a, b)


def gen_mod():
    a = np.array([[7, 10]], dtype=np.int64)
    b = np.array([[3, 4]], dtype=np.int64)
    make_binary_case("mod", "Mod", a, b, dtype=TensorProto.INT64)


def gen_clip():
    x = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
    min_v = np.array(0.0, dtype=np.float32)
    max_v = np.array(1.0, dtype=np.float32)
    node = helper.make_node("Clip", ["X", "Min", "Max"], ["Y"])
    graph = helper.make_graph(
        [node],
        "clip",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("Min", TensorProto.FLOAT, []),
         helper.make_tensor_value_info("Max", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "Min": min_v, "Max": max_v}
    save_test_case("clip", model, inputs, run_ort(model, inputs))


def gen_shape():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    make_unary_case("shape", "Shape", x, out_dtype=TensorProto.INT64)


def gen_reduce_mean():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node("ReduceMean", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph(
        [node], "reduce_mean",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes": axes}
    save_test_case("reduce_mean", model, inputs, run_ort(model, inputs))


def gen_reduce_max():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node("ReduceMax", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph(
        [node], "reduce_max",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes": axes}
    save_test_case("reduce_max", model, inputs, run_ort(model, inputs))


def gen_reduce_sum():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph(
        [node], "reduce_sum",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes": axes}
    save_test_case("reduce_sum", model, inputs, run_ort(model, inputs))


def gen_reduce_prod():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node("ReduceProd", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph(
        [node], "reduce_prod",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes": axes}
    save_test_case("reduce_prod", model, inputs, run_ort(model, inputs))


def gen_reduce_sum_square():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node("ReduceSumSquare", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph(
        [node], "reduce_sum_square",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "axes": axes}
    save_test_case("reduce_sum_square", model, inputs, run_ort(model, inputs))


def gen_split():
    x = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
    split = np.array([3, 5], dtype=np.int64)
    node = helper.make_node("Split", ["X", "split"], ["Y0", "Y1"], axis=1)
    graph = helper.make_graph(
        [node], "split",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8]),
         helper.make_tensor_value_info("split", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y0", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "split": split}
    save_test_case("split", model, inputs, run_ort(model, inputs))


def gen_cast():
    x = np.array([[1.2, -2.8, 3.4]], dtype=np.float32)
    node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT64)
    graph = helper.make_graph(
        [node], "cast",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("cast", model, inputs, run_ort(model, inputs))


def gen_equal():
    a = np.array([[1, 2, 3]], dtype=np.int32)
    b = np.array([[1, 0, 3]], dtype=np.int32)
    make_binary_case("equal", "Equal", a, b, dtype=TensorProto.INT32, out_dtype=TensorProto.BOOL)


def gen_greater():
    a = np.array([[1, 2, 3]], dtype=np.float32)
    b = np.array([[0, 2, 4]], dtype=np.float32)
    make_binary_case("greater", "Greater", a, b, out_dtype=TensorProto.BOOL)


def gen_less():
    a = np.array([[1, 2, 3]], dtype=np.float32)
    b = np.array([[0, 2, 4]], dtype=np.float32)
    make_binary_case("less", "Less", a, b, out_dtype=TensorProto.BOOL)


def gen_where():
    cond = np.array([[True, False, True]])
    x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    node = helper.make_node("Where", ["Cond", "X", "Y"], ["Z"])
    graph = helper.make_graph(
        [node], "where",
        [helper.make_tensor_value_info("Cond", TensorProto.BOOL, [1, 3]),
         helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"Cond": cond, "X": x, "Y": y}
    save_test_case("where", model, inputs, run_ort(model, inputs))


def gen_onehot():
    indices = np.array([0, 2], dtype=np.int64)
    depth = np.array(3, dtype=np.int64)
    values = np.array([0.0, 1.0], dtype=np.float32)
    node = helper.make_node("OneHot", ["indices", "depth", "values"], ["Y"], axis=-1)
    graph = helper.make_graph(
        [node], "onehot",
        [helper.make_tensor_value_info("indices", TensorProto.INT64, [2]),
         helper.make_tensor_value_info("depth", TensorProto.INT64, []),
         helper.make_tensor_value_info("values", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"indices": indices, "depth": depth, "values": values}
    save_test_case("onehot", model, inputs, run_ort(model, inputs))


def gen_expand():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    node = helper.make_node("Expand", ["X", "shape"], ["Y"])
    graph = helper.make_graph(
        [node], "expand",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "shape": shape}
    save_test_case("expand", model, inputs, run_ort(model, inputs))


def gen_slice():
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([1], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    node = helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
    graph = helper.make_graph(
        [node], "slice",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "starts": starts, "ends": ends, "axes": axes}
    save_test_case("slice", model, inputs, run_ort(model, inputs))


def gen_pad():
    x = np.array([1.0, 2.0], dtype=np.float32)
    pads = np.array([1, 1], dtype=np.int64)
    value = np.array(0.0, dtype=np.float32)
    node = helper.make_node("Pad", ["X", "pads", "value"], ["Y"], mode="constant")
    graph = helper.make_graph(
        [node], "pad",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("pads", TensorProto.INT64, [2]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "pads": pads, "value": value}
    save_test_case("pad", model, inputs, run_ort(model, inputs))


def gen_tile():
    x = np.array([1.0, 2.0], dtype=np.float32)
    repeats = np.array([2], dtype=np.int64)
    node = helper.make_node("Tile", ["X", "repeats"], ["Y"])
    graph = helper.make_graph(
        [node], "tile",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("repeats", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "repeats": repeats}
    save_test_case("tile", model, inputs, run_ort(model, inputs))


def gen_topk():
    x = np.array([[1.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    k = np.array([2], dtype=np.int64)
    node = helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=1)
    graph = helper.make_graph(
        [node], "topk",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("K", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Values", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Indices", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "K": k}
    save_test_case("topk", model, inputs, run_ort(model, inputs))


def gen_gather_elements():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices = np.array([[1, 0], [0, 1]], dtype=np.int64)
    node = helper.make_node("GatherElements", ["data", "indices"], ["Y"], axis=1)
    graph = helper.make_graph(
        [node], "gather_elements",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"data": data, "indices": indices}
    save_test_case("gather_elements", model, inputs, run_ort(model, inputs))


def gen_gather_nd():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    node = helper.make_node("GatherND", ["data", "indices"], ["Y"])
    graph = helper.make_graph(
        [node], "gather_nd",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"data": data, "indices": indices}
    save_test_case("gather_nd", model, inputs, run_ort(model, inputs))


def gen_scatter_nd():
    data = np.array([0, 0, 0], dtype=np.int64)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([5, 7], dtype=np.int64)
    node = helper.make_node("ScatterND", ["data", "indices", "updates"], ["Y"])
    graph = helper.make_graph(
        [node], "scatter_nd",
        [helper.make_tensor_value_info("data", TensorProto.INT64, [3]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 1]),
         helper.make_tensor_value_info("updates", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph)
    inputs = {"data": data, "indices": indices, "updates": updates}
    save_test_case("scatter_nd", model, inputs, run_ort(model, inputs))


def gen_argmax():
    x = np.array([[1.0, 3.0], [5.0, 4.0]], dtype=np.float32)
    node = helper.make_node("ArgMax", ["X"], ["Y"], axis=1, keepdims=0)
    graph = helper.make_graph(
        [node], "argmax",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("argmax", model, inputs, run_ort(model, inputs))


def gen_resize():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    node = helper.make_node(
        "Resize", ["X", "", "scales"], ["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [node], "resize",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2]),
         helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "scales": scales}
    save_test_case("resize", model, inputs, run_ort(model, inputs))


def gen_upsample():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    node = helper.make_node("Upsample", ["X", "scales"], ["Y"], mode="nearest")
    graph = helper.make_graph(
        [node], "upsample",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2]),
         helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph, opset=9)
    inputs = {"X": x, "scales": scales}
    save_test_case("upsample", model, inputs, run_ort(model, inputs))


def gen_lrn():
    x = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
    node = helper.make_node("LRN", ["X"], ["Y"], size=3, alpha=0.0001, beta=0.75, bias=1.0)
    graph = helper.make_graph(
        [node], "lrn",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("lrn", model, inputs, run_ort(model, inputs))


def gen_conv_transpose():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    w = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"], strides=[1, 1])
    graph = helper.make_graph(
        [node], "conv_transpose",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2]),
         helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "W": w}
    save_test_case("conv_transpose", model, inputs, run_ort(model, inputs))


def gen_average_pool():
    x = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    node = helper.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2])
    graph = helper.make_graph(
        [node], "average_pool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("average_pool", model, inputs, run_ort(model, inputs))


def gen_range():
    start = np.array(1, dtype=np.int64)
    limit = np.array(5, dtype=np.int64)
    delta = np.array(2, dtype=np.int64)
    node = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
    graph = helper.make_graph(
        [node], "range",
        [helper.make_tensor_value_info("start", TensorProto.INT64, []),
         helper.make_tensor_value_info("limit", TensorProto.INT64, []),
         helper.make_tensor_value_info("delta", TensorProto.INT64, [])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph, opset=11)
    inputs = {"start": start, "limit": limit, "delta": delta}
    save_test_case("range", model, inputs, run_ort(model, inputs))


def gen_cumsum():
    x = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
    graph = helper.make_graph(
        [node], "cumsum",
        [helper.make_tensor_value_info("X", TensorProto.INT32, [3]),
         helper.make_tensor_value_info("axis", TensorProto.INT32, [])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, None)],
    )
    model = make_model_simple(graph, opset=11)
    inputs = {"X": x, "axis": axis}
    save_test_case("cumsum", model, inputs, run_ort(model, inputs))


def gen_not():
    x = np.array([True, False, True])
    make_unary_case("not", "Not", x, opset=OPSET, dtype=TensorProto.BOOL, out_dtype=TensorProto.BOOL)


def gen_size():
    x = np.zeros((2, 3), dtype=np.float32)
    make_unary_case("size", "Size", x, out_dtype=TensorProto.INT64)


def gen_constant_of_shape():
    shape = np.array([2, 1], dtype=np.int64)
    value = helper.make_tensor("value", TensorProto.INT64, [1], [7])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=value)
    graph = helper.make_graph(
        [node], "constant_of_shape",
        [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
    )
    model = make_model_simple(graph)
    inputs = {"shape": shape}
    save_test_case("constant_of_shape", model, inputs, run_ort(model, inputs))


def gen_dropout():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    node = helper.make_node("Dropout", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "dropout",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x}
    save_test_case("dropout", model, inputs, run_ort(model, inputs))


def gen_layer_norm():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    node = helper.make_node("LayerNormalization", ["X", "Scale", "Bias"], ["Y"], axis=-1, epsilon=1e-5)
    graph = helper.make_graph(
        [node], "layer_norm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("Scale", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("Bias", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph, opset=17)
    inputs = {"X": x, "Scale": scale, "Bias": bias}
    save_test_case("layer_norm", model, inputs, run_ort(model, inputs))


def gen_lstm():
    x = np.array([[[1.0]]], dtype=np.float32)
    w = np.ones((1, 4, 1), dtype=np.float32)
    r = np.zeros((1, 4, 1), dtype=np.float32)
    b = np.zeros((1, 8), dtype=np.float32)
    node = helper.make_node("LSTM", ["X", "W", "R", "B"], ["Y", "Y_h", "Y_c"], hidden_size=1)
    graph = helper.make_graph(
        [node], "lstm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1]),
         helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 4, 1]),
         helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 4, 1]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, None)],
    )
    model = make_model_simple(graph)
    inputs = {"X": x, "W": w, "R": r, "B": b}
    save_test_case("lstm", model, inputs, run_ort(model, inputs))


# ── Batch-generated simple op tests ──

def _unary_test(name, op_type, input_data, opset=13, attrs=None):
    np.random.seed(hash(name) % 2**31)
    x = np.array(input_data, dtype=np.float32).reshape(-1)
    node_attrs = {}
    if attrs:
        node_attrs = attrs
    n = helper.make_node(op_type, ["X"], ["Y"], **node_attrs)
    graph = helper.make_graph([n], name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph, opset=opset)
    inputs = {"X": x}
    save_test_case(name, model, inputs, run_ort(model, inputs))

def gen_div():
    np.random.seed(70)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32) + 2.0
    n = helper.make_node("Div", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "div",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])])
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("div", model, inputs, run_ort(model, inputs))

def gen_mul():
    np.random.seed(71)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    n = helper.make_node("Mul", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "mul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])])
    model = make_model_simple(graph)
    inputs = {"A": a, "B": b}
    save_test_case("mul", model, inputs, run_ort(model, inputs))

def gen_tanh_op():
    _unary_test("tanh_op", "Tanh", [-1, 0, 1])

def gen_erf():
    _unary_test("erf", "Erf", [-1, 0, 0.5, 1])

def gen_ceil():
    _unary_test("ceil", "Ceil", [1.3, -2.7, 0.5])

def gen_round_op():
    _unary_test("round_op", "Round", [1.5, 2.5, -1.5])

def gen_sign():
    _unary_test("sign", "Sign", [-3, 0, 5])

def gen_tan():
    _unary_test("tan", "Tan", [0, 0.5, 1.0])

def gen_sinh():
    _unary_test("sinh", "Sinh", [-1, 0, 1])

def gen_cosh():
    _unary_test("cosh", "Cosh", [-1, 0, 1])

def gen_asin():
    _unary_test("asin", "Asin", [-0.5, 0, 0.5])

def gen_acos():
    _unary_test("acos", "Acos", [-0.5, 0, 0.5])

def gen_atan():
    _unary_test("atan", "Atan", [-1, 0, 1])

def gen_flatten():
    np.random.seed(72)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    n = helper.make_node("Flatten", ["X"], ["Y"], axis=1)
    graph = helper.make_graph([n], "flatten",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("flatten", model, {"X": x}, run_ort(model, {"X": x}))

def gen_gelu():
    _unary_test("gelu", "Gelu", [-1, 0, 1], opset=20)

def gen_mish():
    _unary_test("mish", "Mish", [-1, 0, 1], opset=18)

def gen_selu():
    _unary_test("selu", "Selu", [-1, 0, 1])

def gen_softplus():
    _unary_test("softplus", "Softplus", [-1, 0, 1])

def gen_softsign():
    _unary_test("softsign", "Softsign", [-1, 0, 1])

def gen_log_softmax():
    np.random.seed(73)
    x = np.random.randn(1, 5).astype(np.float32)
    n = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=-1)
    graph = helper.make_graph([n], "log_softmax",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])])
    model = make_model_simple(graph)
    save_test_case("log_softmax", model, {"X": x}, run_ort(model, {"X": x}))

def gen_global_max_pool():
    np.random.seed(74)
    x = np.random.randn(1, 2, 3, 3).astype(np.float32)
    n = helper.make_node("GlobalMaxPool", ["X"], ["Y"])
    graph = helper.make_graph([n], "global_max_pool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("global_max_pool", model, {"X": x}, run_ort(model, {"X": x}))

def gen_depth_to_space():
    np.random.seed(75)
    x = np.random.randn(1, 4, 2, 2).astype(np.float32)
    n = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
    graph = helper.make_graph([n], "depth_to_space",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("depth_to_space", model, {"X": x}, run_ort(model, {"X": x}))

def gen_space_to_depth():
    np.random.seed(76)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    n = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
    graph = helper.make_graph([n], "space_to_depth",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("space_to_depth", model, {"X": x}, run_ort(model, {"X": x}))

def gen_reduce_min():
    np.random.seed(77)
    x = np.random.randn(2, 3).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    n = helper.make_node("ReduceMin", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph([n], "reduce_min",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("reduce_min", model, {"X": x, "axes": axes}, run_ort(model, {"X": x, "axes": axes}))

def gen_reduce_l1():
    np.random.seed(78)
    x = np.random.randn(2, 3).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    n = helper.make_node("ReduceL1", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph([n], "reduce_l1",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("reduce_l1", model, {"X": x, "axes": axes}, run_ort(model, {"X": x, "axes": axes}))

def gen_reduce_l2():
    np.random.seed(79)
    x = np.random.randn(2, 3).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    n = helper.make_node("ReduceL2", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph([n], "reduce_l2",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("reduce_l2", model, {"X": x, "axes": axes}, run_ort(model, {"X": x, "axes": axes}))

def gen_greater_or_equal():
    np.random.seed(80)
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([2, 2, 1], dtype=np.float32)
    n = helper.make_node("GreaterOrEqual", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "greater_or_equal",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("greater_or_equal", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_less_or_equal():
    np.random.seed(81)
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([2, 2, 1], dtype=np.float32)
    n = helper.make_node("LessOrEqual", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "less_or_equal",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("less_or_equal", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_and_op():
    a = np.array([True, True, False], dtype=bool)
    b = np.array([True, False, False], dtype=bool)
    n = helper.make_node("And", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "and_op",
        [helper.make_tensor_value_info("A", TensorProto.BOOL, [3]),
         helper.make_tensor_value_info("B", TensorProto.BOOL, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("and_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_or_op():
    a = np.array([True, False, False], dtype=bool)
    b = np.array([False, False, True], dtype=bool)
    n = helper.make_node("Or", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "or_op",
        [helper.make_tensor_value_info("A", TensorProto.BOOL, [3]),
         helper.make_tensor_value_info("B", TensorProto.BOOL, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("or_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_nonzero():
    x = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
    n = helper.make_node("NonZero", ["X"], ["Y"])
    graph = helper.make_graph([n], "nonzero",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("nonzero", model, {"X": x}, run_ort(model, {"X": x}))

def gen_prelu():
    np.random.seed(82)
    x = np.array([-2, -1, 0, 1], dtype=np.float32)
    slope = np.array([0.25], dtype=np.float32)
    n = helper.make_node("PRelu", ["X", "slope"], ["Y"])
    graph = helper.make_graph([n], "prelu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("slope", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])])
    model = make_model_simple(graph)
    save_test_case("prelu", model, {"X": x, "slope": slope}, run_ort(model, {"X": x, "slope": slope}))

def gen_instance_norm():
    np.random.seed(83)
    x = np.random.randn(1, 2, 3, 3).astype(np.float32)
    scale = np.array([1.0, 2.0], dtype=np.float32)
    bias = np.array([0.0, 1.0], dtype=np.float32)
    n = helper.make_node("InstanceNormalization", ["X", "scale", "B"], ["Y"])
    graph = helper.make_graph([n], "instance_norm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 3]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("instance_norm", model, {"X": x, "scale": scale, "B": bias}, run_ort(model, {"X": x, "scale": scale, "B": bias}))

def gen_group_norm():
    np.random.seed(84)
    num_groups = 2
    x = np.random.randn(1, 4, 2, 2).astype(np.float32)
    # ORT expects scale/bias shape = [num_groups], not [num_channels]
    scale = np.ones(num_groups, dtype=np.float32)
    bias = np.zeros(num_groups, dtype=np.float32)
    n = helper.make_node("GroupNormalization", ["X", "scale", "bias"], ["Y"], num_groups=num_groups)
    graph = helper.make_graph([n], "group_norm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2, 2]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [num_groups]),
         helper.make_tensor_value_info("bias", TensorProto.FLOAT, [num_groups])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph, opset=18)
    try:
        save_test_case("group_norm", model, {"X": x, "scale": scale, "bias": bias}, run_ort(model, {"X": x, "scale": scale, "bias": bias}))
    except Exception as e:
        print(f"  SKIP group_norm: {e}")

def gen_grid_sample():
    np.random.seed(85)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    grid = np.random.randn(1, 2, 2, 2).astype(np.float32) * 0.5
    n = helper.make_node("GridSample", ["X", "grid"], ["Y"])
    graph = helper.make_graph([n], "grid_sample",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4]),
         helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph, opset=16)
    save_test_case("grid_sample", model, {"X": x, "grid": grid}, run_ort(model, {"X": x, "grid": grid}))

def gen_scatter_elements():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([[1], [0]], dtype=np.int64)
    updates = np.array([[99], [88]], dtype=np.float32)
    n = helper.make_node("ScatterElements", ["data", "indices", "updates"], ["Y"], axis=1)
    graph = helper.make_graph([n], "scatter_elements",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 1]),
         helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("scatter_elements", model, {"data": data, "indices": indices, "updates": updates},
                   run_ort(model, {"data": data, "indices": indices, "updates": updates}))

def gen_einsum():
    np.random.seed(86)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    n = helper.make_node("Einsum", ["A", "B"], ["Y"], equation="ij,jk->ik")
    graph = helper.make_graph([n], "einsum",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("einsum", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_trilu():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    n = helper.make_node("Trilu", ["X"], ["Y"], upper=1)
    graph = helper.make_graph([n], "trilu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph, opset=14)
    save_test_case("trilu", model, {"X": x}, run_ort(model, {"X": x}))

def gen_max_op():
    a = np.array([1, 4, 3], dtype=np.float32)
    b = np.array([2, 2, 5], dtype=np.float32)
    n = helper.make_node("Max", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "max_op",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])])
    model = make_model_simple(graph)
    save_test_case("max_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_min_op():
    a = np.array([1, 4, 3], dtype=np.float32)
    b = np.array([2, 2, 5], dtype=np.float32)
    n = helper.make_node("Min", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "min_op",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])])
    model = make_model_simple(graph)
    save_test_case("min_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_mean_op():
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([3, 4, 5], dtype=np.float32)
    n = helper.make_node("Mean", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "mean_op",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])])
    model = make_model_simple(graph)
    save_test_case("mean_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_sum_op():
    a = np.array([1, 2], dtype=np.float32)
    b = np.array([3, 4], dtype=np.float32)
    n = helper.make_node("Sum", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "sum_op",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])])
    model = make_model_simple(graph)
    save_test_case("sum_op", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_dequantize_linear():
    x = np.array([0, 128, 255], dtype=np.uint8)
    scale = np.array([0.01], dtype=np.float32)
    zp = np.array([128], dtype=np.uint8)
    n = helper.make_node("DequantizeLinear", ["x", "scale", "zp"], ["Y"])
    graph = helper.make_graph([n], "dequantize_linear",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [3]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("zp", TensorProto.UINT8, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("dequantize_linear", model, {"x": x, "scale": scale, "zp": zp},
                   run_ort(model, {"x": x, "scale": scale, "zp": zp}))

def gen_quantize_linear():
    x = np.array([0, 0.5, 1.0], dtype=np.float32)
    scale = np.array([0.01], dtype=np.float32)
    zp = np.array([0], dtype=np.uint8)
    n = helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["Y"])
    graph = helper.make_graph([n], "quantize_linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("zp", TensorProto.UINT8, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("quantize_linear", model, {"x": x, "scale": scale, "zp": zp},
                   run_ort(model, {"x": x, "scale": scale, "zp": zp}))

def gen_gru():
    np.random.seed(87)
    x = np.array([[[0.1, 0.2]]], dtype=np.float32)
    w = np.ones((1, 6, 2), dtype=np.float32) * 0.01
    r = np.ones((1, 6, 2), dtype=np.float32) * 0.01
    b = np.zeros((1, 12), dtype=np.float32)
    n = helper.make_node("GRU", ["X", "W", "R", "B"], ["Y", "Y_h"], hidden_size=2)
    graph = helper.make_graph([n], "gru",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2]),
         helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 6, 2]),
         helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 6, 2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 12])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("gru", model, {"X": x, "W": w, "R": r, "B": b},
                   run_ort(model, {"X": x, "W": w, "R": r, "B": b}))

def gen_rnn():
    np.random.seed(88)
    x = np.random.randn(1, 1, 2).astype(np.float32)
    w = np.random.randn(1, 1, 2).astype(np.float32) * 0.1
    r = np.random.randn(1, 1, 1).astype(np.float32) * 0.1
    n = helper.make_node("RNN", ["X", "W", "R"], ["Y", "Y_h"], hidden_size=1)
    graph = helper.make_graph([n], "rnn",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2]),
         helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 2]),
         helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
         helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None)])
    model = make_model_simple(graph)
    save_test_case("rnn", model, {"X": x, "W": w, "R": r},
                   run_ort(model, {"X": x, "W": w, "R": r}))

def gen_nms():
    boxes = np.array([[[0,0,1,1],[0,0,1,1],[0,0,0.5,0.5],[2,2,3,3]]], dtype=np.float32)
    scores = np.array([[[0.9, 0.8, 0.7, 0.6]]], dtype=np.float32)
    max_output = np.array([10], dtype=np.int64)
    iou_thresh = np.array([0.5], dtype=np.float32)
    score_thresh = np.array([0.0], dtype=np.float32)
    n = helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_output", "iou_thresh", "score_thresh"], ["Y"])
    graph = helper.make_graph([n], "nms",
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4, 4]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 4]),
         helper.make_tensor_value_info("max_output", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("iou_thresh", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("score_thresh", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    feed = {"boxes": boxes, "scores": scores, "max_output": max_output, "iou_thresh": iou_thresh, "score_thresh": score_thresh}
    save_test_case("nms", model, feed, run_ort(model, feed))

def gen_rms_norm():
    np.random.seed(89)
    x = np.random.randn(1, 3).astype(np.float32)
    scale = np.ones(3, dtype=np.float32)
    n = helper.make_node("SimplifiedLayerNormalization", ["X", "scale"], ["Y"], epsilon=1e-5, axis=-1)
    graph = helper.make_graph([n], "rms_norm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    # RMSNormalization is opset 21; use SimplifiedLayerNormalization as ORT proxy
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 8
    try:
        save_test_case("rms_norm", model, {"X": x, "scale": scale}, run_ort(model, {"X": x, "scale": scale}))
    except Exception:
        pass  # Skip if ORT version doesn't support it


def gen_argmin():
    np.random.seed(90)
    x = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.float32)
    n = helper.make_node("ArgMin", ["X"], ["Y"], axis=1, keepdims=0)
    graph = helper.make_graph([n], "argmin",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("argmin", model, {"X": x}, run_ort(model, {"X": x}))

def gen_isnan():
    x = np.array([1.0, float('nan'), 0.0], dtype=np.float32)
    n = helper.make_node("IsNaN", ["X"], ["Y"])
    graph = helper.make_graph([n], "isnan",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("isnan", model, {"X": x}, run_ort(model, {"X": x}))

def gen_isinf():
    x = np.array([float('inf'), float('-inf'), 0.0], dtype=np.float32)
    n = helper.make_node("IsInf", ["X"], ["Y"])
    graph = helper.make_graph([n], "isinf",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])])
    model = make_model_simple(graph)
    save_test_case("isinf", model, {"X": x}, run_ort(model, {"X": x}))

def gen_castlike():
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    target = np.array([0], dtype=np.int64)
    n = helper.make_node("CastLike", ["X", "target"], ["Y"])
    graph = helper.make_graph([n], "castlike",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("target", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("castlike", model, {"X": x, "target": target}, run_ort(model, {"X": x, "target": target}))


# ── Type variation tests ──
# These test non-float32 type paths through shape/comparison/indexing ops

def gen_transpose_int64():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    n = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])
    graph = helper.make_graph([n], "transpose_int64",
        [helper.make_tensor_value_info("X", TensorProto.INT64, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("transpose_int64", model, {"X": x}, run_ort(model, {"X": x}))

def gen_concat_uint8():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    n = helper.make_node("Concat", ["A", "B"], ["Y"], axis=0)
    graph = helper.make_graph([n], "concat_uint8",
        [helper.make_tensor_value_info("A", TensorProto.UINT8, [2, 2]),
         helper.make_tensor_value_info("B", TensorProto.UINT8, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("concat_uint8", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_where_int32():
    cond = np.array([True, False, True], dtype=bool)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    n = helper.make_node("Where", ["cond", "X", "Y"], ["Z"])
    graph = helper.make_graph([n], "where_int32",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [3]),
         helper.make_tensor_value_info("X", TensorProto.INT32, [3]),
         helper.make_tensor_value_info("Y", TensorProto.INT32, [3])],
        [helper.make_tensor_value_info("Z", TensorProto.INT32, None)])
    model = make_model_simple(graph)
    save_test_case("where_int32", model, {"cond": cond, "X": x, "Y": y},
                   run_ort(model, {"cond": cond, "X": x, "Y": y}))

def gen_gather_uint8():
    data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    indices = np.array([0, 2], dtype=np.int64)
    n = helper.make_node("Gather", ["data", "indices"], ["Y"], axis=1)
    graph = helper.make_graph([n], "gather_uint8",
        [helper.make_tensor_value_info("data", TensorProto.UINT8, [2, 3]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("gather_uint8", model, {"data": data, "indices": indices},
                   run_ort(model, {"data": data, "indices": indices}))

def gen_equal_int64():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 0, 3], dtype=np.int64)
    n = helper.make_node("Equal", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "equal_int64",
        [helper.make_tensor_value_info("A", TensorProto.INT64, [3]),
         helper.make_tensor_value_info("B", TensorProto.INT64, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, None)])
    model = make_model_simple(graph)
    save_test_case("equal_int64", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_reduce_sum_int64():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    n = helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph([n], "reduce_sum_int64",
        [helper.make_tensor_value_info("X", TensorProto.INT64, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("reduce_sum_int64", model, {"X": x, "axes": axes},
                   run_ort(model, {"X": x, "axes": axes}))

def gen_slice_uint8():
    x = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([4], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    n = helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
    graph = helper.make_graph([n], "slice_uint8",
        [helper.make_tensor_value_info("X", TensorProto.UINT8, [5]),
         helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("slice_uint8", model, {"X": x, "starts": starts, "ends": ends, "axes": axes},
                   run_ort(model, {"X": x, "starts": starts, "ends": ends, "axes": axes}))

def gen_pad_uint8():
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pads = np.array([0, 0, 1, 1], dtype=np.int64)
    const_val = np.array([0], dtype=np.uint8)
    n = helper.make_node("Pad", ["X", "pads", "const_val"], ["Y"])
    graph = helper.make_graph([n], "pad_uint8",
        [helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 2]),
         helper.make_tensor_value_info("pads", TensorProto.INT64, [4]),
         helper.make_tensor_value_info("const_val", TensorProto.UINT8, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("pad_uint8", model, {"X": x, "pads": pads, "const_val": const_val},
                   run_ort(model, {"X": x, "pads": pads, "const_val": const_val}))

def gen_cast_int8():
    x = np.array([-1.5, 0, 1.5, 127.9], dtype=np.float32)
    n = helper.make_node("Cast", ["X"], ["Y"], to=3)  # INT8
    graph = helper.make_graph([n], "cast_int8",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.INT8, None)])
    model = make_model_simple(graph)
    save_test_case("cast_int8", model, {"X": x}, run_ort(model, {"X": x}))

def gen_greater_float64():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 1.0], dtype=np.float64)
    n = helper.make_node("Greater", ["A", "B"], ["Y"])
    graph = helper.make_graph([n], "greater_float64",
        [helper.make_tensor_value_info("A", TensorProto.DOUBLE, [3]),
         helper.make_tensor_value_info("B", TensorProto.DOUBLE, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, None)])
    model = make_model_simple(graph)
    save_test_case("greater_float64", model, {"A": a, "B": b}, run_ort(model, {"A": a, "B": b}))

def gen_split_int64():
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    split = np.array([2, 4], dtype=np.int64)
    n = helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=0)
    graph = helper.make_graph([n], "split_int64",
        [helper.make_tensor_value_info("X", TensorProto.INT64, [6]),
         helper.make_tensor_value_info("split", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y1", TensorProto.INT64, None),
         helper.make_tensor_value_info("Y2", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("split_int64", model, {"X": x, "split": split},
                   run_ort(model, {"X": x, "split": split}))

def gen_tile_uint8():
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    repeats = np.array([1, 2], dtype=np.int64)
    n = helper.make_node("Tile", ["X", "repeats"], ["Y"])
    graph = helper.make_graph([n], "tile_uint8",
        [helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 2]),
         helper.make_tensor_value_info("repeats", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)])
    model = make_model_simple(graph)
    save_test_case("tile_uint8", model, {"X": x, "repeats": repeats},
                   run_ort(model, {"X": x, "repeats": repeats}))

def gen_expand_int64():
    x = np.array([[1], [2], [3]], dtype=np.int64)
    shape = np.array([3, 4], dtype=np.int64)
    n = helper.make_node("Expand", ["X", "shape"], ["Y"])
    graph = helper.make_graph([n], "expand_int64",
        [helper.make_tensor_value_info("X", TensorProto.INT64, [3, 1]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)])
    model = make_model_simple(graph)
    save_test_case("expand_int64", model, {"X": x, "shape": shape},
                   run_ort(model, {"X": x, "shape": shape}))


if __name__ == "__main__":
    generators = [
        gen_add_broadcast, gen_sub_div, gen_matmul_2d, gen_gemm_bias,
        gen_relu, gen_sigmoid_tanh, gen_softmax,
        gen_conv2d, gen_maxpool, gen_batchnorm,
        gen_reshape, gen_transpose, gen_globalavgpool,
        gen_concat, gen_gather, gen_squeeze_unsqueeze,
        gen_mlp_small, gen_conv_relu_pool,
        gen_identity, gen_leakyrelu, gen_hardsigmoid, gen_hardswish, gen_elu, gen_log, gen_abs, gen_neg,
        gen_sqrt, gen_exp, gen_floor, gen_sin, gen_cos, gen_reciprocal,
        gen_pow, gen_mod, gen_clip, gen_shape, gen_reduce_mean,
        gen_reduce_max, gen_reduce_sum, gen_reduce_prod, gen_reduce_sum_square, gen_split, gen_cast, gen_equal,
        gen_greater, gen_less, gen_where, gen_onehot, gen_expand, gen_slice, gen_pad,
        gen_tile, gen_topk, gen_gather_elements, gen_gather_nd, gen_scatter_nd,
        gen_argmax, gen_resize, gen_upsample, gen_lrn, gen_conv_transpose,
        gen_average_pool, gen_range, gen_cumsum, gen_not, gen_size,
        gen_constant_of_shape, gen_dropout, gen_layer_norm, gen_lstm,
        # Batch-generated op tests
        gen_div, gen_mul, gen_tanh_op, gen_erf, gen_ceil, gen_round_op, gen_sign,
        gen_tan, gen_sinh, gen_cosh, gen_asin, gen_acos, gen_atan,
        gen_flatten, gen_gelu, gen_mish, gen_selu, gen_softplus, gen_softsign,
        gen_log_softmax, gen_global_max_pool, gen_depth_to_space, gen_space_to_depth,
        gen_reduce_min, gen_reduce_l1, gen_reduce_l2,
        gen_greater_or_equal, gen_less_or_equal, gen_and_op, gen_or_op, gen_nonzero,
        gen_prelu, gen_instance_norm, gen_group_norm, gen_grid_sample,
        gen_scatter_elements, gen_einsum, gen_trilu,
        gen_max_op, gen_min_op, gen_mean_op, gen_sum_op,
        gen_dequantize_linear, gen_quantize_linear,
        gen_gru, gen_rnn, gen_nms, gen_rms_norm,
        # New ops
        gen_argmin, gen_isnan, gen_isinf, gen_castlike,
        # Type variation tests
        gen_transpose_int64, gen_concat_uint8, gen_where_int32, gen_gather_uint8,
        gen_equal_int64, gen_reduce_sum_int64, gen_slice_uint8, gen_pad_uint8,
        gen_cast_int8, gen_greater_float64, gen_split_int64, gen_tile_uint8,
        gen_expand_int64,
    ]
    for gen in generators:
        name = gen.__name__.replace("gen_", "")
        print(f"Generating {name}...")
        gen()
    print(f"Done. Generated {len(generators)} test cases.")
