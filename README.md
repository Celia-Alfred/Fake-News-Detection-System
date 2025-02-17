# Fake News Detection App

Detect Fake News using Machine Learning
This project is a Fake News Classification App built with Streamlit and Machine Learning models to predict whether news content is real or fake. It supports text input, voice input, visualizations, and text-to-speech feedback.

## Features
- Multiple Machine Learning Models (Logistic Regression, Decision Tree, Random Forest)
- Text and Voice Input Support
- Real-Time Predictions
- Visualizations for Model Comparison
- Prediction History Tracking
- User Feedback System
- Preloaded News Data (True.csv, Fake.csv) for Testing

## Project Structure

📁 fake-news-detection
│── app.py                # Streamlit App
│── models                # Pretrained Models
│   │── model_LR.pkl
│   │── model_DT.pkl
│   │── model_RF.pkl
│   │── vectorizer.pkl
│── data                  # Dataset for Testing
│   │── True.csv
│   │── Fake.csv
│── assets                # Additional Resources
│── requirements.txt       # Dependencies
│── README.md             # Project Documentation


## Installation
1. *Clone the Repository:*
   bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   
2. *Install Dependencies:*
   bash
   pip install -r requirements.txt
   
3. *Run the App:*
   bash
   streamlit run app.py
   

## Dataset
- True.csv — Contains real news samples
- Fake.csv — Contains fake news samples
- *Source:* Kaggle Fake News Dataset

## Usage
1. Enter news text manually or use voice input.
2. Click *Predict* to see if the news is real or fake.
3. View model-wise predictions and combined prediction.
4. Check visualizations and prediction history.

## Contributions
Contributions are welcome. Feel free to fork, improve, and submit pull requests.

## License
This project is open-source and available under the MIT License.

Developed by Marie Celia Alfred

