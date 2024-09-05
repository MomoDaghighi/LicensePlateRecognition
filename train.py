from src.data_loader import load_images_and_points
from src.dataset import PlateDataset
from src.model import PlateModel
from src.train import train_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Load data
images, points = load_images_and_points('data/images', 'data/labels')

# Split data into training and validation sets...
# Initialize Dataset and Dataloader
train_dataset = PlateDataset(images_train, points_train)
val_dataset = PlateDataset(images_val, points_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = PlateModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=25)
