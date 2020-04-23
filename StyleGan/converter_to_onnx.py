from model import StyledGenerator
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import onnx
import onnxruntime
from model import StyledGenerator, Discriminator, Generator


generator = StyledGenerator(512).to('cpu')
generator.load_state_dict(torch.load('./stylegan.pt')['g_running'])
generator.eval()

input = torch.randn(1, 512)
torch.onnx.export(generator, input, f="stylegan.onnx", verbose=True)

# Check the model
onnx_model = onnx.load('stylegan.onnx')
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# Inference
session = onnxruntime.InferenceSession('stylegan.onnx', None)
input_name = session.get_inputs()[0].name
print('Input tensor name :', input_name)
x = input.numpy()
outputs = session.run(None, {input_name: x})[0]
print(outputs.shape)
# Output
print('ONNX input\n',x)
print('ONNX output\n',outputs)
