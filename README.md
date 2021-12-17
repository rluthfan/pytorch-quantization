# Pytorch Quantization Module
Part of: [Front End Module](https://github.com/raudipra/front_end_quantization_project) and [Back End Quantization Project](https://github.com/raudipra/back_end_quantization_project).
This repository contains Resnet Quantization implentation in Pytorch, and currently supports Post Training Quantization, and Quantization Aware Training.

## Setup
- `pip install -r requirements.txt`

## ONNX Runtime Serving

### Generate ONNX Model
- `python -m quantization_functions.generate_onnx.py`

### Setup Server
- Clone the repo
- Download the [ONNX models](https://drive.google.com/drive/folders/1-Mc2gVb5yMWstdlm-MKFF3sgPEoFIxne?usp=sharing) and keep it under `pytorch-quantization/checkpoint/onnx/imagenette/`

### Run Server
- `sudo docker pull mcr.microsoft.com/onnxruntime/server`
- `sudo docker run -it -v $(pwd):$(pwd) -p 9001:8001 mcr.microsoft.com/onnxruntime/server --model_path $(pwd)/pytorch-quantization/checkpoint/onnx/imagenette/resnet50_4bit.onnx`
- You can choose the model by changing the `--model_path` argument above.