from src.data_loader import load_images_and_points
from src.dataset import PlateDataset
from src.model import PlateModel
from src.evaluate import evaluate_model
from torch.utils.data import DataLoader

# Load data
images, points = load_images_and_points('data/test_images', 'data/test_labels')
test_dataset = PlateDataset(images, points)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = PlateModel()

# Evaluate the model
evaluate_model(model, test_loader)
