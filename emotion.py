import cv2
from deepface import DeepFace
import time
import requests
from collections import Counter

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize variables to track emotions
emotion_counter = Counter()
start_time = time.time()
duration = 5  # Run for 5 seconds

def capitalize_emotion(emotion):
    """Ensure the emotion is capitalized correctly."""
    return emotion.capitalize()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion and capitalize it
        emotion = capitalize_emotion(result[0]['dominant_emotion'])

        # Update the emotion counter
        emotion_counter[emotion] += 1

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Check if the specified duration has passed
    if time.time() - start_time > duration:
        # Find the most detected emotion and capitalize it
        most_common_emotion = emotion_counter.most_common(1)
        if most_common_emotion:
            emotion = capitalize_emotion(most_common_emotion[0][0])
            print(f"Most detected emotion: {emotion}")
            
            # Send emotion to Flask app via GET request
            response = requests.get(f'http://localhost:8237/callback/create_playlist?mood={emotion}')
            print(f"Response from Flask app: {response.text}")

        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()