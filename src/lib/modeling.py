from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import pickle
import os

# Convert tokenized text back to string format for vectorization
def tokens_to_text(tokens):
    """Convert list of tokens back to text string"""
    if isinstance(tokens, str):
        # If it's already a string, try to evaluate it as a list
        try:
            tokens = eval(tokens)
        except:
            return tokens
    if isinstance(tokens, list):
        return ' '.join(tokens)
    return str(tokens)

def train_model(df, test_size=0.2):
    """
    Train logistic regression model with TF-IDF vectorization
    
    Args:
        df: DataFrame with 'text' and 'target' columns
        test_size: Size of test set (default: 0.2)
    
    Returns:
        tuple: (pipeline, results_dict)
    """
    
    # Prepare the text data - use text_clean if available, otherwise use text
    if 'text_clean' in df.columns:
        text_data = df['text_clean']
    else:
        text_data = df['text'].apply(tokens_to_text)
    
    # Create pipeline with TF-IDF vectorizer and logistic regression
    # Adjust parameters based on dataset size for better handling of small datasets
    dataset_size = len(df)
    if dataset_size < 50:
        # For very small datasets, use more lenient parameters
        min_df = 1
        max_df = 1.0
        max_features = min(1000, dataset_size * 10)
    else:
        # For larger datasets, use more restrictive parameters
        min_df = 2
        max_df = 0.8
        max_features = 5000
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=min_df,  # Adjust based on dataset size
            max_df=max_df  # Adjust based on dataset size
        )),
        ('clf', LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,  # Regularization parameter
            solver='liblinear'  # Good for small datasets
        ))
    ])
    
    # Split the data
    X = text_data
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save model in models directory (use absolute path)
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_filepath = os.path.join(model_dir, 'disaster_tweet_model_trained.pkl')
    
    save_model(pipeline, model_filepath)
    
    return pipeline, results

def save_model(pipeline, filepath):
    """
    Save trained pipeline to disk
    
    Args:
        pipeline: Trained pipeline
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load trained pipeline from disk
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Pipeline: Trained pipeline
    """
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    
    return pipeline

def predict_disaster(text, pipeline):
    """
    Predict if a tweet is about a disaster
    
    Args:
        text: Input text to classify
        pipeline: Trained pipeline
    
    Returns:
        tuple: (prediction, probability)
    """
    # Clean and prepare the text
    text_clean = tokens_to_text(text)
    
    # Make prediction
    prediction = pipeline.predict([text_clean])[0]
    probability = pipeline.predict_proba([text_clean])[0]
    
    return prediction, probability[1]  # Return probability of disaster class
