import cv2
import numpy as np
# Define dictionary and marker parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 200

# Create an empty image to store the full dictionary
dictionary_image = np.zeros((marker_size * 10, marker_size * 25, 3), dtype=np.uint8)

# Loop through all marker IDs and generate individual marker images
for marker_id in range(4):
    marker_image = cv2.aruco.drawMarker(dictionary, marker_id, marker_size)
    # Calculate row and column positions for placement
    row = marker_id // 25
    col = marker_id % 25
    # Paste individual marker image onto the full dictionary image
    marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
    dictionary_image[row * marker_size:(row + 1) * marker_size, col * marker_size:(col + 1) * marker_size] = marker_image

# Save the generated dictionary image
cv2.imwrite("aruco_6x6_250_dictionary.png", dictionary_image)
