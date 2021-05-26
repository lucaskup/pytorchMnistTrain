import torch.nn as nn
import torch
import torch.nn.functional as F

MEAN = 0.1307
STANDARD_DEVIATION = 0.3081


class NetWeb(nn.Module):
    def __init__(self):
        super(NetWeb, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x / 255
        x = (x - MEAN) / STANDARD_DEVIATION
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)


webModel = NetWeb()
webModel.load_state_dict(torch.load('./results/model_web.pth'))
webModel.eval()

dummy_input = torch.zeros(280 * 280 * 4)
torch.onnx.export(webModel, dummy_input,
                  './results/onnx_model.onnx', verbose=True)
