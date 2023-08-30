import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, image_array, transform=None):
        self.data = data
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, sample = self.data[index]
        
        image_with_array = (image + self.image_array).float() # tensor + numpy is tensor

        if self.transform is not None:
            image_with_array = self.transform(image_with_array)

        return image_with_array, sample