import cv2
import numpy as np
from deepface import DeepFace
import pyjokes
import random
import pyttsx3
import streamlit as st

# Load DEEPFACE model
model = DeepFace.build_model('Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to predict emotion
def predict_emotion(frame):
    # Resize frame
    resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the image for DEEPFACE
    img = gray_frame.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Predict emotions using DEEPFACE
    preds = model.predict(img)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]

    return emotion

# Streamlit app
def main():
    st.title("Emotion Detection App")
    st.write("Capture an image from your webcam and detect your emotion!")

    # Add a "Capture Emotion" button
    if st.button("Capture Emotion"):
        # Capture an image using the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # Predict emotion
        emotion = predict_emotion(frame)

        # Convert BGR image to RGB for colorful display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the captured image with the predicted emotion
        st.image(rgb_frame, caption=f"Detected Emotion: {emotion}", use_column_width=True)

        # Define responses based on emotions
        responses = {
            'happy': "HEY! You look happy. Congrats!",
            'sad': pyjokes.get_joke(),
            'angry': random.choice([
                "Take a deep breath and count to ten.",
                "Stay calm and carry on.",
                "Inhale the future, exhale the past.",
                "Keep calm and let it go.",
            ]),
            'neutral': "You seem to be in a neutral mood.",
            'surprise': "You look surprised!",
            'disgust': "You seem disgusted.",
            'fear': "You appear to be fearful."
        }

        # Display the response based on the detected emotion
        if emotion in responses:
            response_text = responses[emotion]
            st.write(f"**Emotion:** {emotion}")
            st.write(f"**Response:** {response_text}")

            # Initialize the text-to-speech engine
            engine = pyttsx3.init()

            # Speak the response
            engine.say(response_text)
            engine.runAndWait()
        else:
            st.warning("Emotion not recognized.")

        # Release the capture
        cap.release()

if __name__ == "__main__":
    main()
