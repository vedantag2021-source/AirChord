"""
Minimal diagnostic: tests ONLY whether OpenCV's window receives keypresses.
No MediaPipe, no complexity -- just a plain window and a key check.

Press ANY key. It should print the key's code. Press 'q' to quit.
"""

import cv2
import numpy as np

# Just a plain gray image, no webcam needed for this test
blank_frame = np.full((300, 500, 3), 100, dtype=np.uint8)
cv2.putText(blank_frame, "Press any key (check terminal)", (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.namedWindow("Key Test", cv2.WINDOW_NORMAL)

print("Window should be open. Click it, then press keys. 'q' to quit.")

while True:
    cv2.imshow("Key Test", blank_frame)
    key = cv2.waitKey(1) & 0xFF

    if key != 255:  # 255 means "no key pressed this frame"
        print(f"Key pressed! Code: {key}, Char: {chr(key) if 32 <= key <= 126 else 'N/A'}")

    if key == ord('q'):
        break

cv2.destroyAllWindows()