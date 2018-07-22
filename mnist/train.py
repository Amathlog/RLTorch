import torch
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l1(x.view(-1, self.input_size)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cv1 = nn.Conv2d(1, 12, 3)
        self.cv2 = nn.Conv2d(12, 24, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(24 * 5 * 5, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.cv1(x)))
        x = self.pool(F.relu(self.cv2(x)))

        x = F.relu(self.l1(x.view(-1, 5 * 5 * 24)))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class MNISTClassifier(object):
    def __init__(self):
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.RandomAffine(45, (0.25, 0.25), (0.5, 1.5), 10),
             transforms.RandomCrop(5, padding=5)]
        )

        self.train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
        self.train_loader = utils.data.DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=1)

        self.test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
        self.test_loader = utils.data.DataLoader(self.test_set, batch_size=32, shuffle=True, num_workers=1)

        # self.net = MLP(28*28)
        self.net = CNN()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def imshow_random(self):
        # get some random training images
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()

        # show images
        img = images[0].numpy().squeeze()
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(img, cmap='Greys')
        plt.title("Number: " + str(labels[0].numpy()))
        plt.show()

    def run_mini_batch(self, data, train=True):
        inputs, labels = data

        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        if train:
            loss.backward()
            self.optimizer.step()

        return loss

    def test(self):
        with torch.no_grad():
            total = 0
            correct = 0
            loss = 0
            for i, data in enumerate(self.test_loader):
                images, labels = data

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                loss += self.criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Error on the test set: {((1 - correct / total) * 100):.3f}%")
        return loss / i

    def train(self, nb_epochs):
        losses = []
        for epoch in range(nb_epochs):
            train_loss = 0
            for i, data in enumerate(self.train_loader, 0):
                train_loss += self.run_mini_batch(data)

            print(f"Train loss after epoch {epoch}: {train_loss / i}")

            test_loss = self.test()

            losses.append((train_loss, test_loss))

        plt.plot(np.array(losses)[0, :])
        plt.plot(np.array(losses)[1, :])
        plt.show()


if __name__ == '__main__':
    classifier = MNISTClassifier()
    print(classifier.net)
    classifier.train(10)