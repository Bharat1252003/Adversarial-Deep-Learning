import torch
import torch.nn as nn
from sklearn.metrics import recall_score as recall
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import time

from config import *
from custom_dataset import CustomDataset

class PopObj:
    def __init__(self, image:np.ndarray):
        self.image = image
        self.fitness = 1e-10

def change(images, alpha):
    return images+alpha.image

def selection(alpha_pop:list[PopObj]):
    W = []
    for a in alpha_pop:
        W.append(a.fitness)

    W = np.array(W)
    min_val = np.min(W)
    max_val = np.max(W)
    W = (W-min_val)/(max_val-min_val)
    W = W/np.sum(W)
    proportion = int(epsilon*len(alpha_pop))
    
    C = np.random.choice(alpha_pop, proportion, replace=True, p=W)
    P1 = np.array([i for i in alpha_pop if i not in C])

    return P1, C

def crossover_img(c1:PopObj, c2:PopObj):
    img_size = c1.image.shape
    slice_size = (img_size[0]*eta_w, img_size[1]*eta_h)
    slice_img_dim = (
        np.random.randint(low=0, high=img_size[0]-slice_size[0]),
        np.random.randint(low=0, high=img_size[1]-slice_size[1])
        )
    
    c1.image[slice_img_dim], c2.image[slice_img_dim] = c2.image[slice_img_dim], c1.image[slice_img_dim]
    
    assert type(c1) == np.ndarray
    assert type(c2) == np.ndarray
    
    return c1, c2

def mutation(m:PopObj):
    step = 0.001*np.random.random()
    mask = np.random.random(m.image.shape) > 0.5
    m.image[mask] += step

    return m

def new_data(train_dataloader, test_dataloader, image_array, batch_size=batch_size):
    output = []

    assert type(image_array) == np.ndarray

    """transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    """

    train_data = train_dataloader.dataset
    test_data = test_dataloader.dataset

    train_dataset = CustomDataset(train_data, image_array)#, transform=transform)
    test_dataset = CustomDataset(test_data, image_array)#, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    output.append((train_dataloader, test_dataloader))
    
    return output

def fitness(data_loader, alpha_pop:list[PopObj], model:nn.Sequential, criterion=nn.NLLLoss()):
    max_fitness = -np.inf
    print("Recall avg for each alpha:")
    for alpha in alpha_pop:
        error_cumul = 0
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1) #Returns the k largest elements of the given input tensor along a given dimension.
                error = recall(labels.view(*top_class.shape), top_class)
                error_cumul += error
        
        error_avg = error_cumul/len(data_loader)
        print(error_avg)

        alpha.fitness = 1 + lamda * (1-error_avg) - np.linalg.norm(alpha.image)

        if alpha.fitness > max_fitness:
            max_fitness = alpha.fitness
            max_a = alpha
    
    return sorted(alpha_pop, key=lambda a: a.fitness, reverse=True)

from PIL import Image
import numpy as np

def save_image(array, name):
    image_array = (array * 255).astype(np.uint8).reshape((100,100,3))
    image = Image.fromarray(image_array)
    image.save(fr'C:\intermediate_images\{name}.jpg')

def save_dataloader_img(train_loader, name):
    for batch in train_loader:
        # Access the first batch of data
        images, labels = batch
        # Select the first image from the batch
        image = images[0]
        break  # Exit the loop after the first batch

    # Convert the image tensor to a NumPy array
    image_array = image.numpy()

    # Convert the image array to the appropriate data type and scale it to the valid range
    image_array = (image_array * 255).astype(np.uint8).reshape((100,100,3))

    # Create a PIL Image object from the array
    image_pil = Image.fromarray(image_array)

    # Save the image
    image_pil.save(fr'C:\intermediate_images\{name}.jpg')
