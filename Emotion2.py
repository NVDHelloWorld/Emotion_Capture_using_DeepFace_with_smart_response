import cv2
import numpy as np
from deepface import DeepFace
import pyjokes
import random
import pyttsx3
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# Load DEEPFACE model
model = DeepFace.build_model(task="facial_attribute" , model_name='Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to preprocess the image for DEEPFACE
def preprocess_image(frame):
    resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)  # Resize frame
    img = resized_frame.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict emotion
def predict_emotion(frame):
    img = preprocess_image(frame)
    preds = model.predict(img)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    return emotion

# Function to get response based on emotion
def get_response(emotion):
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
    return responses.get(emotion, "Emotion not recognized.")
class EmotionDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict emotion
        emotion = predict_emotion(img)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(img, (0, 0), (200, 30), (0, 0, 0), -1)
        cv2.putText(img, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img
# Streamlit app
def main():
    st.title("Emotion Detection App")
    st.write("Capture an image from your webcam and detect your emotion!")

    # Add a sidebar for options
    option = st.sidebar.radio("Select an option", ("Real-time Emotion Detection", "Capture Image from System", "Click Image"))

    if option == "Real-time Emotion Detection":
        """
        st.write("Real-time Emotion Detection is active. Press 'q' to exit.")
        
        # Start capturing video
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open video device.")
            return

        # Initialize image_container with a default image
        default_image = np.zeros((300, 400, 3), dtype=np.uint8)
        image_container = st.image(default_image, caption="Detected Emotion:", use_column_width=True)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            # Predict emotion
            emotion = predict_emotion(frame)

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (0, 0), (200, 30), (0, 0, 0), -1)
            cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the resulting frame in Streamlit
            image_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Detected Emotion: {emotion}", use_column_width=True)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close all windows
        cap.release()
        cv2.destroyAllWindows()
        """
        webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetectionTransformer)

    elif option == "Capture Image from System":
        # Capture an image from the user's system and predict emotion
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to read image.")
                return

            # Predict emotion
            emotion = predict_emotion(image)

            # Convert BGR image to RGB for colorful display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the uploaded image with the predicted emotion
            st.image(rgb_image, caption=f"Detected Emotion: {emotion}", use_column_width=True)

            # Display the response based on the detected emotion
            response_text = get_response(emotion)
            st.write(f"**Emotion:** {emotion}")
            st.write(f"**Response:** {response_text}")

            # Initialize the text-to-speech engine
            engine = pyttsx3.init()

            # Speak the response
            engine.say(response_text)
            engine.runAndWait()
    elif option == "Click Image":
        # Capture an image using the webcam and predict emotion
        if st.button("Click Image"):
            # Capture an image using the webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open video device.")
                return

            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Failed to capture image.")
                return

            # Predict emotion
            emotion = predict_emotion(frame)

            # Convert BGR image to RGB for colorful display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the captured image with the predicted emotion
            st.image(rgb_frame, caption=f"Detected Emotion: {emotion}", use_column_width=True)

            # Display the response based on the detected emotion
            response_text = get_response(emotion)
            st.write(f"**Emotion:** {emotion}")
            st.write(f"**Response:** {response_text}")

            # Initialize the text-to-speech engine
            engine = pyttsx3.init()

            # Speak the response
            engine.say(response_text)
            engine.runAndWait()

if __name__ == "__main__":
    main()
