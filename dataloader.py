# torchvision have Datasets, Transforms and Models specific to Computer Vision
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="runtime",
    train=True,
    download=True,
    transform=ToTensor()
)

print(training_data)

test_data = datasets.FashionMNIST(
    root="runtime",
    train=False,
    download=True,
    transform=ToTensor()
)

print(test_data)

# strange data loader
from torch.utils.data.dataloader import DataLoader

batch_size = 64
training_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in training_dataloader:
    # N, C, H, W
    print(X.shape)
    break
    # 最后不够的时候，取的是所有剩余的