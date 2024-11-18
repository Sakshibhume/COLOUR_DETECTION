
import cv2
import numpy as np
from PIL import Image

# Define color ranges in HSV (Hue, Saturation, Value)
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),  # Red color range (Hue from 0 to 10)
    "Green": ([35, 40, 40], [85, 255, 255]),  # Green color range
    "Blue": ([100, 150, 0], [140, 255, 255]),  # Blue color range
    "Yellow": ([20, 100, 100], [40, 255, 255]),  # Yellow color range
    "Orange": ([5, 150, 150], [15, 255, 255]),  # Orange color range
    "Purple": ([125, 50, 50], [160, 255, 255]),  # Purple color range
    "Pink": ([140, 50, 50], [170, 255, 255]),  # Pink color range
    "White": ([0, 0, 200], [180, 50, 255]),  # White color range
    "Black": ([0, 0, 0], [180, 255, 50])  # Black color range
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Flag to track if any color is detected
    detected_color = None

    # Iterate through all color ranges to detect colors
    for color_name, (lower, upper) in color_ranges.items():
        lowerLimit = np.array(lower)
        upperLimit = np.array(upper)

        # Create a mask for the current color
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        # Find contours of the detected color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Ignore small contours
                # Get the bounding box around the detected object
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Display the color name on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, color_name, (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Set the flag to indicate that a color was detected
                detected_color = color_name

    # Display the frame with bounding boxes and color names
    cv2.imshow('Frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows() 