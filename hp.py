import cv2
import numpy as np
import time

print("Initializing Cloak... Please wait 3 seconds to capture background.")

cap = cv2.VideoCapture(0)

# Allow the camera to warm up
time.sleep(3)

# Capture the background (static frame without cloak)
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis=1)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define cloak color range (red cloak example)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create mask for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Refine mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment out cloak from background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    normal_area = cv2.bitwise_and(img, img, mask=mask_inv)

    # Final output
    final = cv2.addWeighted(cloak_area, 1, normal_area, 1, 0)

    cv2.imshow("Invisibility Cloak", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
