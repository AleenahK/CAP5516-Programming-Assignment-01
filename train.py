import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='torchvision',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
args = vars(parser.parse_args())

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Learning and training parameters.
epochs = 15
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader, test_loader, length_test_data = get_data(batch_size=batch_size)
set
# Define model based on the argument parser string.
if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2).to(device)
    #plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    print('[INFO]: Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=2).to(device) 
    #plot_name = 'resnet_torchvision'
# print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
    
    plot_name = 'RN18_scratch_1.5ALLaugmented_LR0.01_Ep15_'
    model_name = 'models/RN18_scratch_1.5ALLaugmented_LR0.01_Ep15_'   
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )
    
    print('TRAINING COMPLETE')
    
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {correct / length_test_data:.2%}")
    
    #save model
    torch.save(model, model_name)
