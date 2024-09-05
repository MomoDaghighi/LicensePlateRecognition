import os
import cv2
import numpy as np
from .utils import yolo_to_points, augment, resize, normalize_image, normalize_points

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
