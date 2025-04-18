from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class DataTransformation:
    def __init__(self, type_transformation=None):
        self.type_transformation = type_transformation

    def transform_torch(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
        return transform
    
    def transforms_torch_for_huggingface(self, examples):
        transform = self.transform_torch()
        examples_ = [transform(img.convert("RGB")) for img in examples["img"]]
        examples["img"] = examples_
        #examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["img"]]["pixel_values"]
        return examples

    def dataset_to_dataloader(self, dataset, batch_size=4): 
        dataloader = DataLoader(dataset, batch_size)
        return dataloader
    
