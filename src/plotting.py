import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def yolo_to_points(yolo_coords, img_width, img_height):
    points = []
    for i in range(1, len(yolo_coords), 2):  # Skip class label
        x = yolo_coords[i] * img_width
        if i + 1 < len(yolo_coords):
            y = yolo_coords[i + 1] * img_height
            points.append([x, y])
    return np.array(points, dtype=np.float32)


def visualize_points(image, points):
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], c='green', label='True Points')
    plt.title('True Points')
    plt.legend()
    plt.show()


def load_image_and_points(image_path, label_path):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    with open(label_path, 'r') as f:
        yolo_data = [float(num) for num in f.read().strip().split()]

    points = yolo_to_points(yolo_data, img_width, img_height)

    return image, points


def main(image_dir, label_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            if os.path.exists(label_path):
                image, points = load_image_and_points(image_path, label_path)
                visualize_points(image, points)


if __name__ == '__main__':
    image_dir = 'four-corners/images'  # Update this path
    label_dir = 'four-corners/labels'  # Update this path
    main(image_dir, label_dir)
