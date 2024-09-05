import numpy as np
import torch
from .utils import calculate_accuracy, denormalize_points, visualize_results

def evaluate_model(model, test_loader):
    # Load model
    model.load_state_dict(torch.load('models/model_best.pth'))
    model.eval()

    # Evaluation loop
    predicted_points = []
    true_points = []
    original_image_sizes = []  # Store the original sizes of each test image for scaling back
    with torch.no_grad():
        for images, points in test_loader:
            images, points = images.cuda(), points.cuda()
            outputs = model(images)
            predicted_points.append(outputs.cpu().numpy())
            true_points.append(points.cpu().numpy())
            # Assuming the original image sizes were preserved in test_dataset or can be accessed somehow
            original_image_sizes.append([(image.shape[1], image.shape[0]) for image in images.cpu()])

  def calculate_accuracy(true_points, predicted_points, threshold=10):
    true_points = true_points.reshape(-1, 2) * 256
    predicted_points = predicted_points.reshape(-1, 2) * 256
    distances = np.sqrt(np.sum((true_points - predicted_points) ** 2, axis=1))
    accurate_predictions = np.mean(distances < threshold)
    return accurate_predictions

def denormalize_points(points, img_width, img_height):
    points = points.reshape(-1, 2)
    points[:, 0] *= img_width  # scale up x coordinates
    points[:, 1] *= img_height  # scale up y coordinates
    return points.flatten()

if __name__ == '__main__':
    images_dir = '/content/drive/My Drive/four-corners/images'
    labels_dir = '/content/drive/My Drive/four-corners/labels'
    images, points = load_images_and_points(images_dir, labels_dir)

    from sklearn.model_selection import train_test_split

    train_images, test_images, train_points, test_points = train_test_split(images, points, test_size=0.1, random_state=42)
    train_images, val_images, train_points, val_points = train_test_split(train_images, train_points, test_size=0.1, random_state=42)

    train_dataset = PlateDataset(train_images, train_points)
    val_dataset = PlateDataset(val_images, val_points)
    test_dataset = PlateDataset(test_images, test_points)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = PlateModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=5)

    model.load_state_dict(torch.load('/content/drive/My Drive/model_best.pth'))
    model.eval()

    predicted_points = []
    true_points = []
    original_image_sizes = []  # To store the original sizes of each test image for scaling back
    with torch.no_grad():
        for images, points in test_loader:
            images, points = images.cuda(), points.cuda()
            outputs = model(images)
            predicted_points.append(outputs.cpu().numpy())
            true_points.append(points.cpu().numpy())
            # Assuming the original image sizes were preserved in test_dataset or can be accessed somehow
            original_image_sizes.append([(image.shape[1], image.shape[0]) for image in images.cpu()])

    predicted_points = np.vstack(predicted_points)
    true_points = np.vstack(true_points)

    # De-normalize points before calculating MSE
    predicted_points_denorm = []
    true_points_denorm = []
    for (pred_pts, true_pts, (width, height)) in zip(predicted_points, true_points, original_image_sizes):
        predicted_points_denorm.append(denormalize_points(pred_pts, width, height))
        true_points_denorm.append(denormalize_points(true_pts, width, height))

    predicted_points_denorm = np.vstack(predicted_points_denorm)
    true_points_denorm = np.vstack(true_points_denorm)

    mse = ((predicted_points_denorm - true_points_denorm) ** 2).mean()
    print(f"Mean Squared Error: {mse}")

    accuracy = calculate_accuracy(true_points_denorm, predicted_points_denorm)
    print(f"Accuracy (distance < 10 pixels): {accuracy * 100:.2f}%")

    visualize_results(test_images, true_points_denorm, predicted_points_denorm)
