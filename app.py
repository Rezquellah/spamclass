from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vect.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vector = vectorizer.transform([text.lower()])
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    
    predicted_category = 'ham' if prediction[0] == 0 else 'spam'
    probability_ham = probability[0][0]
    probability_spam = probability[0][1]

    return render_template('index.html', 
                           prediction_text=f'Predicted category: {predicted_category}',
                           probability_text=f'Probability of being ham: {probability_ham*100:.2f}%, spam: {probability_spam*100:.2f}%')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

