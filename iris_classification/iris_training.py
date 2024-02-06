"""
from here: https://janakiev.com/blog/pytorch-iris/
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from iris_classification_model import IrisClassificationModel


def pre_process():
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    names = iris["target_names"]
    feature_names = iris["feature_names"]

    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    return train_test_split(X_scaled, y, test_size=0.2, random_state=2)


def train():
    X_train, X_test, y_train, y_test = pre_process()

    model = IrisClassificationModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 100
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()

    loss_list = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

    ax1.plot(accuracy_list)
    ax1.set_ylabel("validation accuracy")
    ax2.plot(loss_list)
    ax2.set_ylabel("validation loss")
    ax2.set_xlabel("epochs")
    # fig.show()

    # save onnx model
    dummy_input = X_test[:1]
    torch.onnx.export(
        model, dummy_input, "iris.onnx", input_names=["input"], output_names=["output"]
    )

    # save torch checkpoint
    torch.save(model, "iris.torch")


if __name__ == "__main__":
    train()
