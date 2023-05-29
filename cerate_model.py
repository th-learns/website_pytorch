import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f'Using {device} device')


# noinspection PyShadowingNames
def get_data():
    training_data = datasets.FashionMNIST(
        root="runtime",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="runtime",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, test_data


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 784 -> 512
            nn.Linear(28 * 28, 512),
            # what's this?
            # x if x > 0, 0 else
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.aa = 11

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack.forward(x)
        return logits


model = NeuralNetwork().to(device)
# why you can print like this?
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# noinspection PyShadowingNames,PyPep8Naming
def train(dataloader, model: nn.Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss: torch.Tensor = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            # what's this?
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


training_data, test_data = get_data()
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-----------------------")
    train(train_dataloader, model, loss_fn, optimizer)