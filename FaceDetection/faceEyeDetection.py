import numpy as np
import cv2

# Load Haar Cascade Classifiers
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# Open Camera with Faster Backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_V4L2 for Linux
cap.set(3, 640)  # Set video width
cap.set(4, 480)  # Set video height

# Skip Initial Frames to Speed Up Startup
for _ in range(5):  
    ret, _ = cap.read()

while True:
    try:
        ret, img = cap.read()
        if not ret:
            print("[ERROR] Camera frame not captured. Exiting...")
            break

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Eye Detection (Improved Settings)
            eyes = eyeCascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.2, 
                minNeighbors=15,  # Increase to reduce false positives
                minSize=(20, 20)  # Ignore very small objects
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('video', img)

        # Reduce frame delay for smoother performance
        if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to quit
            break

    except Exception as e:
        print(f"[ERROR] {e}")  # Print any unexpected errors and continue running

print("\n[INFO] Exiting Program...")
cap.release()
cv2.destroyAllWindows()
