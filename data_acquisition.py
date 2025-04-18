import torch
from torchvision.datasets import CIFAR10
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader

from data_preprocession import DataTransformation


class DataAcquisition:
    def __init__(self, data_path, data_name):
        self.data_path = data_path
        self.name = data_name

    def load_data_local(self, data_path):
        dataset = torch.load(data_path)
        return dataset

    def login_huggingface(self, username, password):
        self.username = username
        self.password = password
        

    def load_data_huggingface(self, data_link_huggingface, split = None):
        if split is not None:
            if split == "train":
                dataset = load_dataset(data_link_huggingface, split = "train")
            elif split == "test":
                dataset = load_dataset(data_link_huggingface, split = "test")
            elif split == "validation":
                dataset = load_dataset(data_link_huggingface, split = "validation")
        else:
            dataset = load_dataset(data_link_huggingface)
        return dataset 

    def load_save_data_huggingface(self, data_path, data_name):
        a=1

    def load_data_torchvision(self, batch_size=4,):
        ###still default implementation
        transform = DataTransformation.transform_torch()
        dataset_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
        dataloader_test = DataLoader(dataloader_test, batch_size=batch_size, shuffle=False, num_workers=2)
        return dataloader_train, dataloader_test
    

if __name__ == "__main__":
    data_link_huggingface = "uoft-cs/cifar10"
    data_acquisition = DataAcquisition(data_path="test",
                                data_name="test")
    dataset_train = data_acquisition.load_data_huggingface(data_link_huggingface, split = "train")
    dataset_test = data_acquisition.load_data_huggingface(data_link_huggingface, split = "test")









