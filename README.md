# pytorch-quantization

## ONNX Runtime Serving

After cloning the repo, in each different server run the command below, change the bit correspondingly
- sudo docker pull mcr.microsoft.com/onnxruntime/server
- sudo docker run -it -v $(pwd):$(pwd) -p 9001:8001 mcr.microsoft.com/onnxruntime/server --model_path $(pwd)/pytorch-quantization/checkpoint/resnet50_4bit.onnx