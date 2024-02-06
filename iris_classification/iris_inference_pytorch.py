import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from iris_classification_model import IrisClassificationModel


def show_iris():
    (data, targets) = load_iris(as_frame=True, return_X_y=True)
    print(targets[targets == 1])


def get_scaled_input(model_input):
    scaler = StandardScaler()
    scaler.fit_transform(load_iris()["data"])
    return scaler.transform(model_input)


def inference():
    # model needs to be in path
    print(IrisClassificationModel.__name__)
    model = torch.load("iris.torch")

    model.eval()
    # Example new data (replace this with actual new data)

    input_0 = [[5.1, 3.5, 1.4, 0.2]]
    input_1 = [[5.5, 2.4, 3.8, 1.1]]
    input_2 = [[6.7, 3.3, 5.7, 2.5]]
    input_data = input_1

    # using the same scaler as when we were training
    scaler = StandardScaler()
    scaler.fit_transform(load_iris()["data"])

    print("input_data", input_data)
    input_data = scaler.transform(input_data)
    print("scaled input_data", input_data)

    # Convert to PyTorch tensor
    # Ideally, load the saved scaler used during training
    input_tensor = torch.tensor(input_data, dtype=torch.float)
    print("input tensor", input_tensor)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)

    print("prediction", prediction)

    # Get the predicted class (use torch.argmax to get the index of the max value)
    predicted_class_index = torch.argmax(prediction, dim=1).item()

    # Optionally, convert this index to the actual class name
    iris = load_iris()
    predicted_class_name = iris["target_names"][predicted_class_index]

    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class name: {predicted_class_name}")


if __name__ == "__main__":
    show_iris()
    inference()
