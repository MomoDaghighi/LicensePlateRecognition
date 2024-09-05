import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import PlateDataset
from .model import PlateModel

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs):
    best_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, points in train_loader:
            images, points = images.to(device), points.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, points)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, points in val_loader:
                images, points = images.to(device), points.to(device)
                outputs = model(images)
                loss = criterion(outputs, points)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/model_best.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model
