import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the trained model
model = load_model('sign_language_model1.h5')

# Define labels for the classes
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

# Function to preprocess the input frame
def preprocess_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to match the input size of the model
    resized = cv2.resize(gray, (48, 48))
    # Normalize the pixel values
    normalized = resized / 255.0
    # Reshape the frame to match the input shape of the model
    preprocessed = np.expand_dims(normalized, axis=0)
    preprocessed = np.expand_dims(preprocessed, axis=-1)
    return preprocessed

# Function to perform inference and display result
def process_frame(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict the sign language gesture
    prediction = model.predict(preprocessed_frame)
    
    # Get the predicted label and accuracy
    predicted_label = labels[np.argmax(prediction)]
    accuracy = np.max(prediction) * 100
    
    # Speak out the predicted label
    engine.say(predicted_label)
    engine.runAndWait()
    
    # Display the predicted label and accuracy on the frame
    text = f'{predicted_label} {accuracy:.2f}%'
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw the rectangular ROI on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

# Define the coordinates of the rectangular region of interest (ROI)
x, y, w, h = 100, 100, 200, 200

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set frame size for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read the first frame to get its dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to capture video.")
    exit()

# Start a separate thread for processing frames
def process_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)

# Start the thread for processing video frames
video_thread = threading.Thread(target=process_video)
video_thread.start()

# Wait for user to press 'q' to exit
while cv2.waitKey(1) & 0xFF != ord('q'):
    continue

# Release the video capture object and join the video processing thread
cap.release()
video_thread.join()

# Close all OpenCV windows
cv2.destroyAllWindows()
