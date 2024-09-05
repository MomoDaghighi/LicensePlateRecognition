import numpy as np
import cv2
import matplotlib.pyplot as plt

def yolo_to_points(yolo_coords, img_width, img_height):
    # Converts YOLO coordinates to points
    points = []
    for i in range(1, len(yolo_coords), 2):  # Skip class label
        x = yolo_coords[i] * img_width
        if i + 1 < len(yolo_coords):
            y = yolo_coords[i + 1] * img_height
            points.append([x, y])
    return np.array(points, dtype=np.float32)

# Augment, Resize, Normalize, and other utility functions...
# (Include all the existing utility functions from your code here)

def visualize_results(images, true_points, predicted_points):
    # Visualize the results of predictions vs actual points
    for i in range(len(images)):
        plt.figure()
        image_uint8 = (images[i] * 255).astype(np.uint8)
        plt.imshow(cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB))
        tp = true_points[i].reshape(-1, 2) * 256
        pp = predicted_points[i].reshape(-1, 2) * 256
        plt.scatter(tp[:, 0], tp[:, 1], c='green', label='True Points')
        plt.scatter(pp[:, 0], pp[:, 1], c='red', label='Predicted Points')
        plt.title('True and Predicted Corners')
        plt.legend()
        plt.show()
