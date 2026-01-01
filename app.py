from flask import Flask, render_template, request
import pickle
import numpy as np
import model  # Importing the feature extraction function

app = Flask(__name__)

# Load the model
try:
    with open('phishing_model.pkl', 'rb') as f:
        clf = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    clf = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        
        if not url:
            return render_template('index.html', prediction_text="Please enter a URL.")

        if clf:
            # Extract features using the function from model.py
            features = np.array([model.extract_features(url)])
            
            # Predict
            prediction = clf.predict(features)
            
            if prediction[0] == 1:
                result = "PHISHING"
                css_class = "danger"
            else:
                result = "LEGITIMATE"
                css_class = "success"
                
            return render_template('index.html', prediction_text=f"Result: {result}", result_class=css_class, url=url)
        else:
            return render_template('index.html', prediction_text="Model not loaded. Please train the model.")

if __name__ == "__main__":
    app.run(debug=True)
