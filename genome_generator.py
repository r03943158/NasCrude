import torch
import torch.nn as nn
import torch.nn.functional as F

class GenomeGenerator(nn.Module):
    def __init__(self, num_layers):
        super(GenomeGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_layers * 3 * 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.shape[0], 3, -1)
        x = self.softmax(x)
        x = torch.max(x, dim=1)[1].float()
        return x

class GenomePredictor(nn.Module):
    def __init__(self, num_layers):
        super(GenomePredictor, self).__init__()
        self.fc1 = nn.Linear(num_layers * 2, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    x = torch.zeros(128, 100)
    generator = GenomeGenerator(12)
    x = generator(x)
    print(x.shape)
    print(x[0])

    print()

    predictor = GenomePredictor(24)
    x = predictor(x)
    print(x.shape)
    print(x[0])
