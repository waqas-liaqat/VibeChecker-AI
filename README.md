# 📢 VibeChecker AI - AI-Powered Tweet Sentiment Analyzer

## 💡 Project Overview
VibeChecker AI is an AI-powered web application that analyzes the sentiment of tweets. It classifies tweets as **Positive 😊** or **Negative 😡** using a **Logistic Regression** model trained on textual data.

---

## 📁 Folder Structure
```
VIBECHECKER AI
│── Artifacts
│   ├── Logistic_Regression.pkl    # Trained model
│   ├── vectorizer.pkl             # TF-IDF Vectorizer
│   ├── raw_data.csv               # Original dataset
│   ├── x_train.csv, y_train.csv   # Training data
│   ├── x_test.csv, y_test.csv     # Testing data
│
│── Assets
│   ├── Main Dataset link.txt      # Link to dataset
│
│── Notebooks
│   ├── EDA_Data_Preprocessing.ipynb  # EDA & Data Preprocessing
│   ├── model_building.ipynb          # Model Training
│
│── app.py                # Streamlit Web App
│── libraries.py          # Required libraries
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── .gitignore            # Ignored files
│── .gitattributes        # Git settings
```

---

## 🚀 Features
✅ **Real-time tweet sentiment analysis**
✅ **Preprocessing of text (stopwords removal, stemming, etc.)**
✅ **User-friendly UI with Streamlit**
✅ **Trained Logistic Regression model for predictions**
✅ **Modular and well-structured code**

---

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/waqas-liaqat.git
cd VibeChecker-AI
```
### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```sh
streamlit run app.py
```

---

## 🎯 How It Works
1️⃣ Enter a tweet in the input field.
2️⃣ Click **"Analyze Sentiment"**.
3️⃣ The model will process the text and display whether the tweet is **Positive 😊** or **Negative 😡**.

---

## 📊 Model & Training Details
- Models: **Logistic Regression** & **MultinomialNB**
- Feature Extraction: **TF-IDF Vectorization**
- Dataset: Available in `Assets/Main Dataset link.txt`
- Notebook: Model training and preprocessing steps can be found in `Notebooks/`

---

## 🔥 Future Enhancements
- 🏆 **Support for neutral sentiment**
- 🎯 **Deployment on cloud platforms**
- 🌍 **Multilingual sentiment analysis**

---

## 📌 Credits
Made with ❤️ by **Muhammad Waqas**. Feel free to contribute or reach out for collaboration! 🚀
Dataset is available on kaggle: [Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
---

## ⭐ Show Some Love
If you like this project, please ⭐ **star** this repository! 😊

