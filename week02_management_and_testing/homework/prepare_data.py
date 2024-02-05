import sys

from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    train_dataset = CIFAR10(sys.argv[1], download=True, train=True)
