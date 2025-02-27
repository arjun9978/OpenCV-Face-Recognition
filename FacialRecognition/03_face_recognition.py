import cv2
import numpy as np

# Load the recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']  # Ensure these match the IDs in your dataset

# Initialize the webcam
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Adjust the confidence threshold if necessary
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(f"Detected: {id} with confidence {confidence}")  # Print the name and confidence
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print(f"Unknown face detected with confidence {confidence}")  # Print for unknown faces

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cam.release()
cv2.destroyAllWindows()
