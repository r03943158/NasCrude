import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from decode_model import *
from train_s_steps import *
from genome_generator import *

batch_size_data = 512

epochs = 100
batch_size = 16
num_layers_search = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_shape = np.array([1, 3, 32, 32])
num_classes = 10
writer = SummaryWriter()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_data, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

generator = GenomeGenerator(num_layers_search).to(device)
predictor = GenomePredictor(num_layers_search).to(device)
optimizer_G = optim.SGD(generator.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_P = optim.SGD(predictor.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

log_f = open("log.txt","w+")

for epoch in range(epochs):
    optimizer_P.zero_grad()
    z = np.random.normal(0, 1, (batch_size, 100))
    z = torch.from_numpy(z).float().to(device)
    model_genomes = generator(z)
    predicted_acc = predictor(model_genomes)
    predicted_acc = predicted_acc.view(-1)
    real_acc = []
    for i in range(model_genomes.shape[0]):
        temp_acc = train_s_steps(4, DecodedModel(model_genomes[i], image_shape, num_classes), trainloader, testloader, device)
        real_acc.append(temp_acc)
    average_acc = sum(real_acc) / len(real_acc)
    real_acc = torch.from_numpy(np.array(real_acc)).float().to(device)
    loss_predictor = torch.abs(torch.sum(predicted_acc - real_acc)) * 10000
    loss_predictor.backward()
    optimizer_P.step()

    optimizer_G.zero_grad()
    z = np.random.normal(0, 1, (batch_size, 100))
    z = torch.from_numpy(z).float().to(device)
    model_genomes = generator(z)
    predicted_acc = predictor(model_genomes)
    loss_generator = torch.sum(1 - predicted_acc) * 100
    loss_generator.backward()
    optimizer_G.step()

    print("Epoch: {}, loss_generator: {}, loss_predictor: {}, average_acc".format(epoch,
            loss_generator, loss_predictor, average_acc))

    writer.add_scalar('loss_generator', loss_generator, epoch)
    writer.add_scalar('loss_predictor', loss_predictor, epoch)
    writer.add_scalar('average_acc', average_acc, epoch)

    torch.save(generator.state_dict(), "./models/generator{}.pth".format(epoch))
    torch.save(predictor.state_dict(), "./models/predictor{}.pth".format(epoch))
writer.close()
