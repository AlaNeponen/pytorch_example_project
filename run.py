import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import autoencoder
import cnn

trainingData = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

testData = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

trainingDataLoader = DataLoader(trainingData, batch_size=64, shuffle=True)
testDataLoader = DataLoader(testData, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

def trainModel(dataloader, model, lossFn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediciton made by the model, and the loss measured by using the loss function
        pred = model(X)
        loss = lossFn(pred, y)

        # Reset the gradients for this iteration
        optimizer.zero_grad()
        
        # Backpropagate the prediction loss
        loss.backward()

        # Adjust the parameters with the optimizer
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"Current loss: {loss}, Progress: [{current}/{size}]")

def testModel(dataloader, model, lossFn):
    size = len(dataloader.dataset)
    numBatches = len(dataloader)
    testLoss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            testLoss += lossFn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    testLoss /= numBatches
    correct /= size
    print(f"Avg Test Loss: {testLoss}, Accuracy: {(100*correct)}")
    return testLoss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    
    def earlyStop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Train and test first model

model_1 = cnn.Cnn(numChannels=1, classes=len(trainingData.classes)).to(device)

lossFunction_1 = nn.CrossEntropyLoss()
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.001)

stopper = EarlyStopper(patience=3, min_delta=0.005)


epochs = 25
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-- -- -- -- --")
    trainModel(trainingDataLoader, model_1, lossFunction_1, optimizer_1)
    validationLoss = testModel(testDataLoader, model_1, lossFunction_1)
    if stopper.earlyStop(validationLoss):
        print("\nEarly stopping triggered")
        break

print("\nModel 1 trained succesfully")
torch.save(model_1.state_dict(), "model_1_weights.pth")


# Train autoencoder and plot testset images with reconstructed images for comparison

model_2 = autoencoder.Autoencoder().to(device)

lossFunction_2 = nn.BCELoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=0.001)
outputs = []

epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-- -- -- -- --")

    for (image, _) in trainingDataLoader:
        image = image.to(device)
        decoded = model_2(image)
        loss = lossFunction_2(decoded, image)

        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()

    print(f"Loss:{loss.item()}")
    outputs.append((t, image, decoded))

print("\nModel 2 trained succesfully")
torch.save(model_2.state_dict(), "model_2_weights.pth")

for k in range(0, epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    images = outputs[k][1].cpu().detach().numpy()
    decodedImages = outputs[k][2].cpu().detach().numpy()
    
    for i, item in enumerate(images):
        if i>= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(decodedImages):
        if i>= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])
plt.show()


# Test classifier success % on images produced by autoencoder

size = len(testDataLoader.dataset)
numBatches = len(testDataLoader)
testLoss, correct = 0, 0

with torch.no_grad():
    for X, y in testDataLoader:
        X, y = X.to(device), y.to(device)
        reconstructed = model_2(X)
        pred = model_1(reconstructed)
        testLoss += lossFunction_1(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

testLoss /= numBatches
correct /= size
print(f"\nAvg Test Loss In Cross-Model Test: {testLoss}, Accuracy: {(100*correct)}")