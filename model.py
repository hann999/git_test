import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(300)  # 批量标准化。
        self.fc1 = nn.Linear(300, 128)
        self.tanh = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        y = x
        return y

if __name__ == '__main__':
    TTR_model = Net()
    input = torch.ones(10000,300)
    output = TTR_model(input)
    print(output.shape)
    print(TTR_model.parameters())