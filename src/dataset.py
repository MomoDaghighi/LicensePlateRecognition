import torch
from torch.utils.data import Dataset

class PlateDataset(Dataset):
    def __init__(self, images, points):
        self.images = images
        self.points = points

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        points = self.points[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        points = torch.tensor(points, dtype=torch.float32)
        return image, points
