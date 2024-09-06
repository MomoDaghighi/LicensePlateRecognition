import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


def yolo_to_points(yolo_coords, img_width, img_height):
    points = []
    for i in range(1, len(yolo_coords), 2):  # Skip class label
        x = yolo_coords[i] * img_width
        if i + 1 < len(yolo_coords):
            y = yolo_coords[i + 1] * img_height
            points.append([x, y])
    return np.array(points, dtype=np.float32)


def augment(image, points):
    h, w = image.shape[:2]
    points = points.reshape(-1, 2)

    # Small Rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        points = M.dot(points_ones.T).T

    # Small Translation
    if np.random.rand() > 0.5:
        tx = np.random.uniform(-0.03 * w, 0.03 * w)
        ty = np.random.uniform(-0.03 * h, 0.03 * h)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
        points += [tx, ty]

    # Small Zoom
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        image = cv2.warpAffine(image, M, (w, h))
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        points = np.dot(M, points.T).T[:, :2]

    # Gaussian blur with a small kernel size
    if np.random.rand() > 0.5:
        ksize = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Random Contrast Adjustment
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    # Random Brightness Adjustment
    if np.random.rand() > 0.5:
        beta = np.random.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=1, beta=beta)

    # Small Perspective Change
    if np.random.rand() > 0.5:
        dst_points = points + np.random.uniform(-3, 3, points.shape)
        M = cv2.getPerspectiveTransform(points.astype(np.float32), dst_points.astype(np.float32))
        image = cv2.warpPerspective(image, M, (w, h))
        points = cv2.perspectiveTransform(points[None, :, :], M)[0]

    # Ensure points remain within image boundaries
    points = np.clip(points, [0, 0], [w - 1, h - 1])

    return image, points.flatten()


def resize(image, points, new_size=(256, 256)):
    old_size = image.shape[:2]
    image = cv2.resize(image, new_size)
    points = points.reshape(-1, 2)
    points[:, 0] *= new_size[1] / old_size[1]
    points[:, 1] *= new_size[0] / old_size[0]
    return image, points.flatten()


def normalize_image(image):
    return image / 255.0


def normalize_points(points, img_width, img_height):
    points = points.reshape(-1, 2)
    points[:, 0] /= img_width
    points[:, 1] /= img_height
    return points.flatten()


def load_images_and_points(image_dir, label_dir):
    images = []
    points = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))
            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                with open(label_path, 'r') as f:
                    yolo_data = [float(num) for num in f.read().strip().split()]
                if len(yolo_data) != 9:
                    continue
                img_height, img_width = image.shape[:2]
                point_array = yolo_to_points(yolo_data, img_width, img_height)
                image, point_array = augment(image, point_array)
                image, point_array = resize(image, point_array)
                image = normalize_image(image)
                point_array = normalize_points(point_array, 256, 256)
                images.append(image)
                points.append(point_array.flatten())
    return np.array(images), np.array(points)


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
