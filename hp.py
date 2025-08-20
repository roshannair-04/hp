# hp.py — Blue Invisibility Cloak (with optional HSV calibration)
import cv2
import numpy as np
import time

# --- defaults tuned for light/mid blue towel ---
LOWER = np.array([85, 50, 50])     # LH, LS, LV
UPPER = np.array([130, 255, 255])  # UH, US, UV

def create_calib_window():
    cv2.namedWindow("Calibrate HSV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate HSV", 480, 220)
    for name, val, maxv in [
        ("LH", int(LOWER[0]), 179), ("UH", int(UPPER[0]), 179),
        ("LS", int(LOWER[1]), 255), ("US", int(UPPER[1]), 255),
        ("LV", int(LOWER[2]), 255), ("UV", int(UPPER[2]), 255),
    ]:
        cv2.createTrackbar(name, "Calibrate HSV", val, maxv, lambda x: None)

def read_calib_bounds():
    lh = cv2.getTrackbarPos("LH", "Calibrate HSV")
    uh = cv2.getTrackbarPos("UH", "Calibrate HSV")
    ls = cv2.getTrackbarPos("LS", "Calibrate HSV")
    us = cv2.getTrackbarPos("US", "Calibrate HSV")
    lv = cv2.getTrackbarPos("LV", "Calibrate HSV")
    uv = cv2.getTrackbarPos("UV", "Calibrate HSV")
    return np.array([lh, ls, lv]), np.array([uh, us, uv])

def capture_background(cap, num_frames=30, flip=True):
    time.sleep(1)  # tiny warmup
    bg = None
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok: continue
        if flip: frame = cv2.flip(frame, 1)
        bg = frame
    return bg

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    flip = True
    print("Capturing background... (move out of frame)")
    background = capture_background(cap, num_frames=40, flip=flip)
    if background is None:
        print("Failed to capture background.")
        return

    calib = False  # calibration window off by default

    while True:
        ok, frame = cap.read()
        if not ok: break
        if flip: frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # read HSV from sliders if calibration is on
        if calib:
            lower, upper = read_calib_bounds()
        else:
            lower, upper = LOWER, UPPER

        # mask for blue cloak
        mask = cv2.inRange(hsv, lower, upper)

        # clean mask: open then dilate
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        inv = cv2.bitwise_not(mask)

        # keep non-cloak region from current frame
        keep = cv2.bitwise_and(frame, frame, mask=inv)
        # take cloak region from background
        cloak = cv2.bitwise_and(background, background, mask=mask)

        out = cv2.addWeighted(keep, 1, cloak, 1, 0)

        # UI text
        cv2.putText(out, "HP Cloak: blue | b=recap bg  c=calibrate  q=quit",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Invisibility Cloak (Blue)", out)

        if calib:
            # show helper views in calibration
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('b'):
            print("Re-capturing background... move out of frame.")
            background = capture_background(cap, num_frames=40, flip=flip)
        elif key == ord('c'):
            calib = not calib
            if calib:
                print("Calibration ON: adjust sliders until ONLY the cloak is white.")
                create_calib_window()
            else:
                print("Calibration OFF.")
                try:
                    cv2.destroyWindow("Calibrate HSV")
                    cv2.destroyWindow("Mask")
                except: pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
