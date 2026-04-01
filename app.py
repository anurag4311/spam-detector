from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    text = clean_text(text)   
    data = vectorizer.transform([text]).toarray()
    result = model.predict(data)[0]

    if result == 1:
        output = "Spam / Fake News ❌"
    else:
        output = "Real / Not Spam ✅"

    return render_template('index.html', prediction_text=output)

# Chatbot API
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message'].lower()

    if "spam" in user_msg:
        reply = "Spam messages are unwanted or harmful messages."
    elif "fake news" in user_msg:
        reply = "Fake news is false or misleading information."
    elif "model" in user_msg:
        reply = "This model uses Machine Learning with TF-IDF."
    elif "accuracy" in user_msg:
        reply = "The model has around 90-96% accuracy."
    elif "hello" in user_msg:
        reply = "Hello! How can I help you?"
    elif "hi" in user_msg:
        reply = "Hi there! Ask me anything about spam detection."
    elif "how are you" in user_msg:
        reply = "I am an AI, always ready to help you!"
    elif "bye" in user_msg:
        reply = "Goodbye! Have a great day."
    elif "who are you" in user_msg:
        reply = "I am an AI chatbot integrated into this project."
    else:
        reply = "I can answer questions about spam, fake news, and this project."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)