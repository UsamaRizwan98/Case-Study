import os
import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, data_dir="./data/cifar10", num_workers=2):
    """
    Loads CIFAR-10 from local data_dir (dataset must be pre-downloaded).
    """
    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
        raise FileNotFoundError(
            f"CIFAR-10 dataset not found in {data_dir}. "
            f"Run `python download_cifar10.py` first."
        )

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Train set
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform_train
    )

    # Test set
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return trainloader, testloader
