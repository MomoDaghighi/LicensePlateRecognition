import cv2
import numpy as np
import os
from extract import extract
from masking import yolo_to_points


def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Extract the license plate image using the points
    plate_image = extract(image, points)

    # Apply Gaussian blur to the plate image
    blurred_plate = cv2.GaussianBlur(plate_image, (101, 101), 0)

    # Compute the size of the bounding box for the plate
    width = int(max(points[:, 0]) - min(points[:, 0]))
    height = int(max(points[:, 1]) - min(points[:, 1]))

    # Resize the blurred plate to fit the original plate's bounding box
    blurred_plate_resized = cv2.resize(blurred_plate, (width, height))

    # Get the perspective transform matrix
    src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src_points, points)

    # Warp the blurred plate to fit the license plate area
    warped_blurred_plate = cv2.warpPerspective(blurred_plate_resized, transform_matrix,
                                               (image.shape[1], image.shape[0]))

    # Create a mask to blend the warped blurred plate into the original image
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.astype(int), (255, 255, 255))

    # Blend the warped blurred plate with the original image
    result = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    result = cv2.bitwise_or(result, warped_blurred_plate)

    return result


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

        # Apply the blur to the image
        result_image = blur(image, points)

        # Save the resulting image
        result_image_path = os.path.join('results3', image_name)  # Save to a results directory
        os.makedirs('results3', exist_ok=True)
        cv2.imwrite(result_image_path, result_image)
