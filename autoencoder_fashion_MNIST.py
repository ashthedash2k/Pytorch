import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.transforms import transforms

#transform data to pytorch tensors
transforms = transforms.ToTensor()

fashion_data = datasets.FashionMNIST(root='./data',  download=True, transform=transforms) #train=True,

data_loader = torch.utils.data.DataLoader(fashion_data, batch_size=64, shuffle=True)

#iterating through our data
dataiter = iter(data_loader)
images, labels = dataiter.next()

#output will get the minimum tensor and the maximum tensor in the dataset --> important for our last activation
print(torch.min(images), torch.max(images))

class autoencoder(nn.Module):
    def __init__(self, epochs=10, batchSize=128, learningRate=1e-3, weight_decay=1e-5):
        super(autoencoder, self).__init__()
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weight_decay = weight_decay

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # reduces from n * 724 to 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # cause tensors are 0, 1
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

    def train(self):
        for epoch in range(self.epochs):
            for data in data_loader:
                img, _ = data
                img = img.view(img.size(0), -1)
                img = Variable(img)

                #predict
                output = self(img)

                # find loss
                loss = self.loss(output, img)

                # perform back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'epoch {epoch + 1}/{self.epochs}, loss: {loss.data:.4f}')



model = autoencoder()
model.train()
