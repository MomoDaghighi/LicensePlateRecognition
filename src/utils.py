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
        dst_points = points + np.random.uniform(-1, 1, points.shape)
        M = cv2.getPerspectiveTransform(points.astype(np.float32), dst_points.astype(np.float32))
        image = cv2.warpPerspective(image, M, (w, h))
        points = cv2.perspectiveTransform(points[None, :, :], M)[0]

    # Ensure points remain within image boundaries
    points = np.clip(points, [0, 0], [w - 1, h - 1])

    return image, points.flatten()


def resize(image, points, new_size=(256, 256)):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(new_size[0]) / max(old_size)
    new_size_unpadded = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size_unpadded[1], new_size_unpadded[0]))

    # Calculate padding
    delta_w = new_size[1] - new_size_unpadded[1]
    delta_h = new_size[0] - new_size_unpadded[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Update points
    points = points.reshape(-1, 2)
    points *= ratio
    points += [left, top]
    return image, points.flatten()



def normalize_image(image):
    return image / 255.0


def normalize_points(points, img_width, img_height):
    points = points.reshape(-1, 2)
    points[:, 0] /= img_width
    points[:, 1] /= img_height
    return points.flatten()

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
