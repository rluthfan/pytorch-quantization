import os
import pathlib

import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .post_training_quant_model import QResnet18, QResnet50

def to_numpy(tensor):
	if tensor.requires_grad:
		return tensor.detach().cpu().numpy()
	return tensor.cpu().numpy()

MODELS = ["resnet18", "resnet50"]
PATH = "checkpoint/imagenette_"
OUTPUT_PATH = "checkpoint/onnx/imagenette/"
N_CLASS = 10

batch_size = 1
dummy_image = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
input_names = ["input"]
output_names = ["output"]


pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

for model_arc in MODELS:
	for subdir in os.listdir(PATH + model_arc):
		subdir_path = os.path.join(PATH + model_arc, subdir)
		print(subdir_path, os.path.isdir(subdir_path))
		if os.path.isdir(subdir_path):
			model = None
			onnx_path = None
			if subdir == "base_model":
				if model_arc == "resnet18":
					model = torchvision.models.resnet18(pretrained=False)
				else:
					model = torchvision.models.resnet50(pretrained=False)
				# Change top layer
				model.fc = nn.Linear(model.fc.in_features, N_CLASS)
				model.load_state_dict(torch.load(subdir_path + '/model_weights.pt',
												 map_location=torch.device('cpu')))
				model.eval()
				onnx_path = OUTPUT_PATH + model_arc + "_32bit.onnx"
			else:
				if model_arc == "resnet18":
					model = QResnet18(num_class=10)
				else:
					model = QResnet50(num_class=10)
				model.load_state_dict(torch.load(subdir_path + '/model_weights_quantized.pt', 
												 map_location=torch.device('cpu')))
				model.eval()
				onnx_path = OUTPUT_PATH + model_arc + "_" + subdir[3:] + ".onnx"
			
			torch.onnx.export(
			    model, 
			    dummy_image,
			    onnx_path,
			    verbose=False,
			    input_names=input_names,
			    output_names=output_names,
			    export_params=True,
			    dynamic_axes={
			        'input' : { 0 : 'batch_size' },    # variable length axes
			        'output' : { 0 : 'batch_size' }
			    },
			    opset_version=11
			)

			# Check onnx model
			onnx_model = onnx.load(onnx_path)
			onnx.checker.check_model(onnx_model)

			# compare ONNX Runtime and PyTorch results
			torch_out = model(dummy_image)
			ort_session = onnxruntime.InferenceSession(onnx_path)

			# compute ONNX Runtime output prediction
			ort_inputs = {
				ort_session.get_inputs()[0].name: to_numpy(dummy_image)
			}
			ort_outs = ort_session.run(None, ort_inputs)
			try:
				np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], 
										   rtol=1e-03, atol=1e-05)
			except Exception as e:
				print(e)
				print("Output difference too far on {}".format(onnx_path))