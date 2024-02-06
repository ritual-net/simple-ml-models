import numpy
import onnx
import onnxruntime as ort
import torch

from iris_inference_pytorch import get_scaled_input


def main():
    onnx_model = onnx.load("iris.onnx")
    onnx.checker.check_model(onnx_model)

    input_0 = numpy.array([[5.1, 3.5, 1.4, 0.2]])
    input_1 = numpy.array([[5.5, 2.4, 3.8, 1.1]])
    input_2 = numpy.array([[6.7, 3.3, 5.7, 2.5]])
    input_scaled = get_scaled_input(input_2)
    print("input scaled: ", input_scaled)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float).numpy()
    print(f"input_tensor: {input_tensor}")
    ort_sess = ort.InferenceSession("iris.onnx")
    outputs = ort_sess.run([], {"input": input_tensor})
    # Print Result
    result = outputs[0].argmax(axis=1)
    print(f"result: {result}")


def check_input_shape():
    model = onnx.load("iris.onnx")
    inputs = {}
    for inp in model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    print(f"inputs: {inputs}")
    print(f"shape: {shape}")


if __name__ == "__main__":
    main()
