import torch
from torch import nn
from torchvision import models

from config import *

def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #print(model)

    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # new layer for this model
    model.fc = nn.Sequential(
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model

def train_loop(train_loader, test_loader, optimizer, criterion, hyper_params = 0, model:nn.Sequential=create_model()):
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1) #Returns the k largest elements of the given input tensor along a given dimension.
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))
                print(
                    f"Epoch {epoch+1}/{epochs} .. "
                    f"Train loss: {running_loss/print_every:.3f} .. "
                    f"Test loss: {test_loss/len(test_loader):.3f} .. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f} .. "
                )
                
                running_loss = 0
                model.train()
    
    return model
"""
from torch import optim
from train_test_split import train_test_split
#train_loader, test_loader = train_test_split(data_dir, 0.2)
#rint(train_loader.dataset.classes)
model = create_model()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

train_loop(train_loader, model=model, optimizer=optimizer, criterion=criterion)

torch.save(model.state_dict(), r'C:\Bharat\College\Sem6\ADL_Project\attempt3\models\model.pth')
"""