import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load data
data = pd.read_csv("spam.csv", encoding='latin-1')
data.columns = ['label', 'message']

# Convert label
data['label'] = data['label'].map({'ham':0, 'spam':1})

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

data['message'] = data['message'].apply(clean_text)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(data['message']).toarray()
y = data['label']

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print("Model saved successfully!")