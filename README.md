##Fake News Detection App##
Detect Fake News using Machine Learning
This project is a Fake News Classification App built with Streamlit and Machine Learning models to predict whether news content is Real or Fake. It supports text input, voice input, visualizations, and text-to-speech feedback.

 ##Features##
- Multiple Machine Learning Models (Logistic Regression, Decision Tree, Random Forest)
 Text & Voice Input Support
 Real-Time Predictions
 Visualizations for Model Comparison
 Prediction History Tracking
 User Feedback System
 Preloaded News Data (True.csv, Fake.csv) for Testing

📂 Project Structure
📁 fake-news-detection
│── 📄 app.py                # Streamlit App
│── 📂 models                # Pretrained Models
│   │── model_LR.pkl
│   │── model_DT.pkl
│   │── model_RF.pkl
│   │── vectorizer.pkl
│── 📂 data                  # Dataset for Testing
│   │── True.csv
│   │── Fake.csv
│── 📂 assets                # Additional Resources
│── 📄 requirements.txt       # Dependencies
│── 📄 README.md             # Project Documentation

🛠 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the App
streamlit run app.py

📊 Dataset
True.csv → Contains real news samples
Fake.csv → Contains fake news samples
Source: Kaggle Fake News Dataset

🎤 Usage
1️⃣ Enter news text manually or use voice input
2️⃣ Click Predict to see if the news is Real or Fake
3️⃣ View model-wise predictions & combined prediction
4️⃣ Check visualizations & prediction history

🌟 Contributions
Feel free to fork, improve, and submit PRs!

📜 License
This project is open-source and available under the MIT License.

✨ Developed with ❤️ by Marie Celia Alfred

