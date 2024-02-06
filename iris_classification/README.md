# Iris Classification
This repo contains the code for a simple iris classification model. The code for 
training the model is written following [this tutorial](https://janakiev.com/blog/pytorch-iris/).

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python iris_training.py
```
This generates two files: `iris.onnx` & `iris.torch`

### Running Inference From Pytorch
```bash
python iris_inference_pytorch.py
```

### Running Inference From ONNX
```bash
python iris_inference_onnx.py
```
