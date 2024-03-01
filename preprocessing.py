import cv2
import numpy as np
def load_and_process_rgb_image(image_path):
    # Load the image in RGB format
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Convert RGB to grayscale
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Resize the image to match the input size expected by the VGG16 model
    resized_image = cv2.resize(grayscale_image, (28, 28))

    # Expand dimensions to match the expected input shape (batch size, height, width, channels)
    processed_image = np.expand_dims(resized_image, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image
