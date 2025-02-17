import streamlit as st
import pickle
import re
import pandas as pd
import string
import plotly.graph_objects as go
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from datetime import datetime

# Load models and vectorizer
with open(r'C:\Users\alfre\OneDrive\Desktop\lang-jupi\lang-jupi\model_LR.pkl', 'rb') as file:
    LR = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\lang-jupi\lang-jupi\model_DT.pkl', 'rb') as file:
    DT = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\lang-jupi\lang-jupi\model_RF.pkl', 'rb') as file:
    RF = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\lang-jupi\lang-jupi\vectorizer.pkl', 'rb') as file:
    vectorization = pickle.load(file)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function for making predictions
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return pred_LR[0], pred_DT[0], pred_RF[0]

# Convert prediction output
def output_label(n):
    return "Real News" if n == 1 else "Fake News"

# Combined voting mechanism
def combined_prediction(predictions):
    return 1 if sum(predictions) >= 2 else 0

# Text-to-speech conversion
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Speech-to-text recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak into the microphone.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("There was an error with the speech recognition service.")
        return ""

# Streamlit UI setup
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title('📰 Fake News Classification App')
st.markdown("""
Welcome to our Fake News Classification App. This tool predicts whether a given news content is Fake or Real.
You can enter the news content manually or use voice input for predictions.
""")

# Initialize session state variables
if 'pred_LR' not in st.session_state:
    st.session_state.pred_LR = None
    st.session_state.pred_DT = None
    st.session_state.pred_RF = None
    st.session_state.combined_pred = None
    st.session_state.history = []

# Sidebar: Model selection
st.sidebar.header("Model Selection")
use_LR = st.sidebar.checkbox("Use Logistic Regression", value=True)
use_DT = st.sidebar.checkbox("Use Decision Tree", value=True)
use_RF = st.sidebar.checkbox("Use Random Forest", value=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Prediction", "📊 Visualizations", "📜 Prediction History", "📝 User Feedback"])

# Prediction Tab
with tab1:
    st.subheader("Enter News Content")
    sentence = st.text_area("Type or paste the news content here:", "", height=200)

    if st.button("Record Voice Input 🎤"):
        sentence = recognize_speech()

    predict_btt = st.button("Predict 🧠")

    if predict_btt or sentence:
        if sentence:
            with st.spinner('Analyzing...'):
                st.session_state.pred_LR, st.session_state.pred_DT, st.session_state.pred_RF = manual_testing(sentence)
                st.session_state.combined_pred = combined_prediction(
                    [st.session_state.pred_LR, st.session_state.pred_DT, st.session_state.pred_RF]
                )

            result = output_label(st.session_state.combined_pred)
            st.subheader("Combined Prediction")
            st.metric("Overall Prediction", result, "🧠")

            # Convert result to speech
            audio_bytes = text_to_speech(f"The combined prediction is {result}.")
            st.audio(audio_bytes, format="audio/mp3")

            st.success("Prediction complete! See the combined result above.")

            # Store in history
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'news': sentence,
                'predictions': {
                    'Logistic Regression': st.session_state.pred_LR,
                    'Decision Tree': st.session_state.pred_DT,
                    'Random Forest': st.session_state.pred_RF,
                    'Combined': st.session_state.combined_pred
                }
            })
        else:
            st.warning("Please enter news content for prediction.")

# Visualization Tab
with tab2:
    st.subheader("Model Prediction Visualizations")
    if st.session_state.pred_LR is not None:
        fig = go.Figure()
        
        if use_LR and st.session_state.pred_LR is not None:
            fig.add_trace(go.Bar(name='Logistic Regression', x=['Prediction'], y=[st.session_state.pred_LR], marker_color='blue'))
        if use_DT and st.session_state.pred_DT is not None:
            fig.add_trace(go.Bar(name='Decision Tree', x=['Prediction'], y=[st.session_state.pred_DT], marker_color='green'))
        if use_RF and st.session_state.pred_RF is not None:
            fig.add_trace(go.Bar(name='Random Forest', x=['Prediction'], y=[st.session_state.pred_RF], marker_color='orange'))

        fig.update_layout(title="Model Predictions", xaxis_title="Models", yaxis_title="Prediction (0 = Fake, 1 = Real)", barmode='group')

        st.plotly_chart(fig)
    else:
        st.info("Run a prediction first to see the visualizations.")

# Prediction History Tab
with tab3:
    st.subheader("Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.write("### History")
        st.dataframe(history_df)
    else:
        st.info("No prediction history available.")

# User Feedback Tab
with tab4:
    st.subheader("User Feedback")

    feedback_form = st.form(key="feedback_form")
    with feedback_form:
        feedback_text = st.text_area("Please provide your feedback on the predictions:", "", height=100)
        feedback_submit = st.form_submit_button("Submit Feedback")

    if feedback_submit:
        if feedback_text:
            if 'feedback' not in st.session_state:
                st.session_state.feedback = []

            st.session_state.feedback.append({
                'timestamp': datetime.now(),
                'feedback': feedback_text
            })

            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter feedback before submitting.")