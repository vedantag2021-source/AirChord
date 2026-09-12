"""
Phase 1: Webcam Capture Test
------------------------------
Goal: Confirm we can open the webcam and display a live video feed.
Press 'q' to quit the window.
"""

import cv2

def main():
    # 0 refers to the default webcam. If you have multiple cameras,
    # you might need to try 1, 2, etc.
    cap = cv2.VideoCapture(0)

    # Check if the webcam actually opened successfully
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # cap.read() returns two things:
        # ret -> True/False, whether the frame was captured successfully
        # frame -> the actual image (as a NumPy array) from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip the frame horizontally so it acts like a mirror
        # (feels natural when looking at yourself on screen)
        frame = cv2.flip(frame, 1)

        # Display the frame in a window titled "AirChord - Webcam Test"
        cv2.imshow("AirChord - Webcam Test", frame)

        # Wait 1ms for a key press. If 'q' is pressed, break the loop.
        # 0xFF is a mask to correctly capture the key regardless of platform.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release the webcam resource and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()