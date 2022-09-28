import onnx
import torch


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.W = torch.nn.Parameter(torch.empty(2, 1))

    def forward(self, X):
        return torch.matmul(X, self.W)


model = NeuralNet()
X = torch.empty(3, 2)
torch.onnx.export(
    model, X, "testdata/matmul_1_ir_13.onnx", verbose=True, input_names=["X"], output_names=["Y"], opset_version=13
)
