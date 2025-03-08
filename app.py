import re
import pickle
import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ✅ Set Page Title
st.set_page_config(page_title="AI-Powered Tweet Sentiment Analyzer", page_icon="💬", layout="centered")

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Initialize lemmatizer and stopwords
stem = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))
stopwords=set(stopwords.words("english"))
# ✅ Text Preprocessing Function
def text_preprocessor(tweet):
    # Removes mentioned person
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove URLs
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower() 
    # Remove special characters
    tweet = re.sub(r"[^a-zA-Z0-9\s]", '', tweet)
    # Split into words, remove stop words, and stem
    words = [stem.stem(word) for word in tweet.split() if word not in stopwords] 
    # Join words back into a single string (correct this part)
    tweet = ' '.join(words)
    return tweet

# ✅ Load Vectorizer & Model
@st.cache_data
def load_vectorizer():
    with open("Artifacts/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_model():
    with open("Artifacts\Logistic Regression.pkl", "rb") as f:
        return pickle.load(f)

vectorizer = load_vectorizer()
model = load_model()

# ✅ UI - Title & Description
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">💬 AI-Powered Tweet Sentiment Analyzer</h1>
    <p style="text-align: center; font-size: 18px;">Enter a tweet and find out its sentiment!</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ✅ Input Field for Tweet
tweet_input = st.text_area("📝 Enter Tweet", help="Type a tweet to analyze its sentiment.", height=150)

# ✅ Predict Sentiment Function
def predict_sentiment(tweet):
    df = pd.DataFrame([tweet], columns=["tweet"])
    df["tweet"] = df["tweet"].apply(text_preprocessor)
    transformed_text = vectorizer.transform(df["tweet"])
    prediction = model.predict(transformed_text)
    return prediction[0]

# ✅ Prediction Button
if st.button("Analyze Sentiment", use_container_width=True):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a tweet.")
    else:
        sentiment = predict_sentiment(tweet_input)
        
        if sentiment == 1:
            st.success("😊 Sentiment: Positive")
        else:
            st.warning("😡 Sentiment: Negative")

# ✅ Footer
st.markdown(
    "<div style='text-align: center; padding: 10px;'>Made By **Muhammad Waqas** using Streamlit & Machine Learning</div>", 
    unsafe_allow_html=True
)
