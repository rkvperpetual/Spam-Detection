from flask import Flask, request, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input message from user
    input_sms = request.form.get('message')

    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize input
    vector_input = tfidf.transform([transformed_sms])

    # Convert to dense if needed
    vector_input = vector_input.toarray()  # Ensure dense format for SVC

    # Predict
    result = model.predict(vector_input)[0]

    # Return result and input message to the web page
    if result == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"

    return render_template('index.html', prediction=prediction, input_sms=input_sms)

if __name__ == '__main__':
    app.run(debug=True)
