import cv2
import numpy as np
import os


def yolo_to_points(yolo_coords, img_width, img_height):
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(yolo_coords), 2):
        x = yolo_coords[i] * img_width
        y = yolo_coords[i + 1] * img_height
        points.append([x, y])
    return np.array(points, dtype=np.float32)


def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Define the width and height of the output image with a fixed aspect ratio
    plate_width = 450
    plate_height = int(plate_width / 4.5)

    # Define the destination points for the perspective transform
    dst_points = np.array([
        [0, 0],
        [plate_width - 1, 0],
        [plate_width - 1, plate_height - 1],
        [0, plate_height - 1]
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(points, dst_points)

    # Apply the perspective transformation to get the corrected plate image
    plate_image = cv2.warpPerspective(image, transform_matrix, (plate_width, plate_height))

    return plate_image


# Directory containing the images and YOLO labels
images_dir = 'images/'
labels_dir = 'labels/'

# Process each image in the images directory
for image_name in os.listdir(images_dir):
    # Load the image
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Load the corresponding YOLO label file
    label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Parse the YOLO label file
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        yolo_coords = [float(x) for x in parts[1:]]

        # Convert YOLO coordinates to corner points
        points = yolo_to_points(yolo_coords, img_width, img_height)

        # Extract the license plate from the image
        plate_image = extract(image, points)

        # Save the extracted license plate image
        result_image_path = os.path.join('results2', f'extracted_{os.path.splitext(image_name)[0]}.jpg')
        os.makedirs('results2', exist_ok=True)
        cv2.imwrite(result_image_path, plate_image)
