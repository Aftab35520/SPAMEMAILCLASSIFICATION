from flask import Flask, jsonify, render_template, request
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Download stopwords and load data
nltk.download('stopwords')
df = pd.read_csv("./spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocessing setup
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
stop_words = set(stopwords.words("english"))
tfidf_vectorizer = TfidfVectorizer()

def preprocess_email(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Prepare data and train model
df["clean_text"] = df["text"].apply(preprocess_email)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

def predict_email(email):
    cleaned_email = preprocess_email(email)
    email_vectorized = tfidf_vectorizer.transform([cleaned_email])
    prediction = model.predict(email_vectorized)[0]
    return prediction == 1  # Returns True for spam, False for ham

# Flask routes
@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    is_spam = predict_email(text) 
    return jsonify({
    'text': text[:100] + '...' if len(text) > 100 else text,
    'is_spam': bool(is_spam),  
    'label': 'Spam' if is_spam else 'Not Spam'
})


if __name__ == '__main__':
    app.run(debug=True)