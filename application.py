import json
import cv2
import numpy as np
import pyttsx3
import streamlit as st
import sqlite3
from datetime import datetime
from difflib import get_close_matches
from textblob import TextBlob
from keras.models import load_model
import requests
import os

PORT = int(os.environ.get('PORT', 8501))  # Use the provided port or a default port


# URLs for required files
emotion_model_url = "https://mycompanion.s3.ap-south-1.amazonaws.com/emotion_model.hdf5"
haarcascade_url = "https://mycompanion.s3.ap-south-1.amazonaws.com/haarcascade_frontalface_default.xml"
knowledge_base_url = "https://mycompanion.s3.ap-south-1.amazonaws.com/knowledge_base.json"

# Function to download files from URLs
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Download the emotion recognition model
emotion_model_path = 'emotion_model.hdf5'
download_file(emotion_model_url, emotion_model_path)

# Download the haarcascade file
haarcascade_path = 'haarcascade_frontalface_default.xml'
download_file(haarcascade_url, haarcascade_path)

# Download the knowledge base JSON file
knowledge_base_path = 'knowledge_base.json'
download_file(knowledge_base_url, knowledge_base_path)

# Load the knowledge base
with open(knowledge_base_path, 'r') as knowledge_file:
    knowledge_base = json.load(knowledge_file)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the emotion recognition model
emotion_model = load_model(emotion_model_path)

# Function to load knowledge base from a JSON file
def load_kb(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

# Function to find the best matching question in the knowledge base
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None


# Function to analyze the sentiment of a text
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_polarity = analysis.sentiment.polarity
    
    if sentiment_polarity < -0.2:
        return "Your text seems to be negative. Is there anything I can help you?"
    elif sentiment_polarity > 0.2:
        return "YOur text seems to be Positive. Is there anything you want to share"
    else:
        return "uhmmm....Well lets talk something else"

# Function for text-to-speech conversion
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to detect emotion in a frame
def detect_emotion(frame):
    # Preprocess the frame and extract the face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Replace with the actual path
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "Neutral"  # Default to neutral if no face is detected

    (x, y, w, h) = faces[0]
    face = gray[y:y + h, x:x + w]

    # Normalize and resize the face image to (64, 64)
    face = cv2.resize(face, (64, 64))

    # Expand dimensions to match model input shape
    face = np.expand_dims(face, axis=-1)  # Add a channel dimension (grayscale)

    # Normalize the pixel values to be in the range [0, 1]
    face = face / 255.0

    # Use the emotion recognition model to predict the emotion
    emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

    # Determine the most likely emotion from the prediction
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    detected_emotion_index = np.argmax(emotion_prediction)
    detected_emotion = emotion_labels[detected_emotion_index]

    return detected_emotion

# Function to map emotion to a chatbot response
def map_emotion_to_response(detected_emotion):
    if detected_emotion == 'Happy':
        return "I sense you're happy and I'm glad you're feeling happy!"
    elif detected_emotion == 'Sad':
        return "I sense you're sad but I'm here to chat if you want to talk about it."
    elif detected_emotion == 'Angry':
        return "I sense you're angry but let's just try to stay calm and positive."
    elif detected_emotion == 'Surprise':
        return "Wooo what made you look surprised?"
    elif detected_emotion == 'Fear':
        return "Relax dear, just let me know the reason why you're afraid."
    elif detected_emotion == 'Disgust':
        return "Oh, what happened? Tell me more."
    else:
        return "How can I assist you today?"

# Function to create the SQLite database for chat history
def create_database():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a chat message into the database
def insert_chat_message(user_message, bot_message):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (user_message, bot_message)
        VALUES (?, ?)
    ''', (user_message, bot_message))
    conn.commit()
    conn.close()

# Function to retrieve chat history for a specific date
def get_chat_history(date):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        SELECT user_message, bot_message, timestamp
        FROM chat_history
        WHERE DATE(timestamp) = DATE(?)
    ''', (date,))
    history = c.fetchall()
    conn.close()
    return history

def get_answer(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if isinstance(q["question"], list):
            # Handle cases where there are multiple valid questions
            if question.lower() in [q.lower() for q in q["question"]]:
                return q["answer"]
        elif q["question"].lower() == question.lower():
            return q["answer"]
    return None

# Create the Streamlit app
def main():
    st.title('My Companion')

    # Load the knowledge base
    knowledge_base = load_kb('knowledge_base.json')

    # Create a user input field
    user_input = st.text_area("You:")

    if st.button('Send') and user_input:
        user_message = "You: " + user_input
        st.text(user_message)
        insert_chat_message(user_message, '')

        if user_input.lower() == 'quit':
            bot_message = "Bot: Goodbye!"
            st.text(bot_message)
            text_to_speech("Goodbye!")
            insert_chat_message('', bot_message)
        else:
            # Check if the user's input matches any question in the knowledge base
            best_match = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])
            if best_match:
                bot_message = "Bot: " + get_answer(best_match, knowledge_base)
            else:
                sentiment_response = analyze_sentiment(user_input)
                bot_message = "Bot: " + sentiment_response

                # Capture and process webcam frames for emotion detection
                ret, frame = cap.read()
                detected_emotion = detect_emotion(frame)

                bot_message = "Bot: " + map_emotion_to_response(detected_emotion)

            st.text(bot_message)
            text_to_speech(bot_message)
            insert_chat_message('', bot_message)

    # History sidebar
    st.sidebar.title("Chat History")
    selected_date = st.sidebar.date_input("Select Date", datetime.today())
    if st.sidebar.button("Load History"):
        history = get_chat_history(selected_date)
        st.sidebar.text("Chat History for " + selected_date.strftime('%Y-%m-%d'))
        for entry in history:
            st.sidebar.text(entry[0])
            st.sidebar.text(entry[1])

# Create the SQLite database (run this only once)
create_database()

# Run the Streamlit app
if __name__ == "__main__":
    main()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
