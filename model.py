import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Feature Extraction Function
def extract_features(url):
    """
    Extracts features from a given URL or text.
    Features:
    - Length of URL
    - Count of '@'
    - Count of '-'
    - Count of '.'
    - Presence of 'https'
    - Presence of suspicious keywords
    """
    features = []
    
    # Feature 1: Length
    features.append(len(url))
    
    # Feature 2: Count of '@'
    features.append(url.count('@'))
    
    # Feature 3: Count of '-'
    features.append(url.count('-'))
    
    # Feature 4: Count of '.'
    features.append(url.count('.'))
    
    # Feature 5: HTTPS usage (1 if yes, 0 if no)
    features.append(1 if "https" in url else 0)
    
    # Feature 6: Suspicious keywords
    keywords = ['login', 'verify', 'update', 'free', 'urgent', 'secure', 'account']
    keyword_count = sum(1 for keyword in keywords if keyword in url.lower())
    features.append(keyword_count)
    
    return features

# 2. Main Training Process
if __name__ == "__main__":
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        print("Error: dataset.csv not found. Please ensure it exists in the same directory.")
        exit()

    print("Extracting features...")
    # Apply feature extraction to the 'text' column
    # Use a list comprehension for efficiency
    X = np.array([extract_features(url) for url in df['text']])
    y = df['label']

    # 3. Split Dataset
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("Training Logistic Regression model...")
    # Using Logistic Regression as requested (Random Forest is also an option)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Model
    print("Saving model to 'phishing_model.pkl'...")
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")
