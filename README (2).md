
# 📧 Spam Detection Using Machine Learning

This project implements a machine learning-based spam detection system for SMS messages. It uses natural language processing (NLP) techniques and classification algorithms to distinguish between spam and ham (non-spam) messages. The system is deployed as a web app using Flask.

## 🚀 Features

- Data cleaning and preprocessing
- Text vectorization using TF-IDF
- Model training with classification algorithms
- Flask-based web interface for real-time prediction
- Serialized model and vectorizer for fast inference

## 🗂️ Project Structure

```
Spam-Detection/
├── app.py                      # Flask web app
├── sms-spam-detection.ipynb    # Model training and analysis
├── spam.csv                    # Dataset used
├── model.pkl                   # Trained model
├── vectorizer.pkl              # TF-IDF vectorizer
└── templates/
    └── index.html              # Frontend HTML template
```

## 📊 Dataset

- Dataset: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- It consists of 5,574 labeled SMS messages (spam or ham).

## 🧠 Model Training

- Preprocessing: Lowercasing, punctuation removal, stopword removal, stemming
- Feature Extraction: TF-IDF Vectorizer
- Model Used: Multinomial Naive Bayes (or similar)
- Model and vectorizer are saved using `pickle` for deployment

## 🧪 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/rkvperpetual/Spam-Detection.git
cd Spam-Detection
```

2. **(Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Flask app**

```bash
python app.py
```

5. **Open the app**

Visit `http://127.0.0.1:5000` in your browser.

## 🧾 Example

- **Input:** "Congratulations! You won a $1000 Walmart gift card. Click to claim."
- **Prediction:** Spam

## 🛠 Requirements

- Python 3.7+
- Flask
- pandas
- scikit-learn
- nltk

> Tip: Use `nltk.download('stopwords')` and `nltk.download('punkt')` before running preprocessing.

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo, make improvements, and submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
