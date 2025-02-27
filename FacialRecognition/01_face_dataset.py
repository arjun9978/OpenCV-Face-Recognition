import cv2
import os
import time

# Create 'dataset' folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Faster camera startup on Windows
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Flush initial frames to reduce warm-up delay
for i in range(5):  
    ret, _ = cam.read()

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Enter user id and press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait...")

count = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Camera frame not captured. Exiting...")
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)  # Reduced scale factor for faster detection

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save face image in 'dataset' folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

        # Introduce a small delay between captures to give you time to adjust your position
        time.sleep(1)  # 1-second delay between capturing images

    # Optional: Display countdown or prompt
    print(f"Captured {count} images, please adjust your position.")
    
    # Stop capturing when either 30 images are taken or the ESC key is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27 or count >= 30:  # ESC key or 30 images captured
        break

print("\n [INFO] Exiting Program and cleanup...")
cam.release()
cv2.destroyAllWindows()
