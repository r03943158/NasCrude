import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import *

class DecodedModel(nn.Module):
    def __init__(self, genome, image_shape, num_classes):
        super(DecodedModel, self).__init__()
        #image_shape should be [b, c, h, w], genome should be [even number]
        assert genome.shape[0] % 2 == 0
        self.genome = genome
        self.module_list = nn.ModuleList()
        self.current_scale = 32
        self.current_channel = 64
        self.num_classes = num_classes

        self.initial_conv = SameRes(image_shape[1], 64)

        next_channel = -1
        for layer in range(genome.shape[0] // 2):
            if genome[layer * 2 + 1] == 0 and self.current_channel != 64:
                next_channel = self.current_channel // 2
            elif genome[layer * 2 + 1] == 2 and self.current_channel != 512:
                next_channel = self.current_channel * 2
            else:
                next_channel = self.current_channel

            if genome[layer * 2] == 0 and self.current_scale != 2:
                self.module_list.append(DownSample(self.current_channel, next_channel))
                self.current_scale = self.current_scale // 2
            elif genome[layer * 2] == 2 and self.current_scale != 32:
                self.module_list.append(UpSample(self.current_channel, next_channel))
                self.current_scale = self.current_scale * 2
            else:
                self.module_list.append(SameRes(self.current_channel, next_channel))
                next_scale = 1

            self.current_channel = next_channel

        x = torch.zeros(1, image_shape[1], image_shape[2], image_shape[3])
        compressed_size = self._get_conv_size(x)

        self.fc1 = nn.Linear(compressed_size, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def _get_conv_size(self, x):
        x = self.initial_conv(x)
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.initial_conv(x)
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    genome = np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0])
    image_shape = np.array([128, 3, 32, 32])
    num_classes = 10
    model = DecodedModel(genome, image_shape, num_classes)
    x = torch.zeros(32, image_shape[1], image_shape[2], image_shape[3])
    x = model(x)
    print(x.shape)
