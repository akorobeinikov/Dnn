from model import StyledGenerator
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import onnx
import onnxruntime
from torchvision import utils
from model import StyledGenerator, Discriminator, Generator

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style




# Check the model
onnx_model = onnx.load('stylegan.onnx')
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# Inference
session = onnxruntime.InferenceSession('stylegan.onnx')
input_name = session.get_inputs()[0].name
print('Input tensor name :', input_name)
input = torch.randn(1, 512)
x = input.numpy()

generator = StyledGenerator(512).to('cpu')
generator.load_state_dict(torch.load('./stylegan.pt')['g_running'])
generator.eval()
outputs = session.run(None, {input_name: x})[0]
mean_style = get_mean_style(generator, 'cpu')
outputs2 = generator(
        torch.randn(1, 512).to('cpu'),
        step=8,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
utils.save_image(torch.Tensor(outputs), 'sample_onnx.png', nrow=1, normalize=True, range=(-1, 1))
print(outputs.shape)
# Output
print('ONNX input\n',x)
print('ONNX output\n',outputs)
print('ONNX output\n',outputs2)
