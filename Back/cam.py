import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Define emotion categories
categories = ['happy', 'angry', 'neutral', 'sad', 'surprise']

# Load the trained model
model_path = './model/keypoint_classifier/keypoint_classifier.hdf5'
model = tf.keras.models.load_model(model_path)

# Set up Mediapipe modules for face detection and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Define function to extract facial landmarks
def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return [(landmark.x, landmark.y) for landmark in landmarks]

        return None

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Extract facial landmarks from the frame
    landmarks = extract_landmarks(frame)

    if landmarks:
        # Normalize the landmarks' values
        landmarks = np.array(landmarks)
        landmarks = (landmarks - landmarks.min(axis=0)) / (landmarks.max(axis=0) - landmarks.min(axis=0))

        # Reshape the input to match the model's expected shape
        input_data = np.expand_dims(landmarks.flatten(), axis=0)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions)
        predicted_emotion = categories[predicted_class_index]

        # Display the predicted emotion on the frame
        cv2.putText(frame, predicted_emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) == 27:
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()