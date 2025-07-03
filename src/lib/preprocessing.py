# Improved version of clean_text function
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pandas as pd
from nltk.corpus import stopwords
import logging

def clean_text(text):
    """
    Improved text cleaning function with stemming and better preprocessing
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        return ""
    
    if pd.isna(text) or not text.strip():
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags (but keep the text after #)
    text = re.sub(r'@\w+', '', text)  # Remove mentions completely
    text = re.sub(r'#', '', text)    # Remove # but keep the word
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize stemmer
    stemmer = PorterStemmer()
    
    # Filter and stem
    cleaned_tokens = []
    for token in tokens:
        # Skip if stopword or too short
        if token not in stop_words and len(token) > 2:
            # Apply stemming
            stemmed_token = stemmer.stem(token)
            cleaned_tokens.append(stemmed_token)
    
    return ' '.join(cleaned_tokens)

def preprocess_dataframe(df, text_column='text'):
    """
    Apply text cleaning to a DataFrame column
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{text_column}' column")

    # Log the start of preprocessing
    logging.info("Cleaning text data")
    
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Apply the clean_text function to create a cleaned text column
    result_df['text_clean'] = result_df[text_column].apply(clean_text)
    
    # Remove rows where text_clean is empty (from null/invalid inputs)
    result_df = result_df[result_df['text_clean'].str.len() > 0]
    
    # Update the 'text' column if it's different from the text_column
    if text_column != 'text':
        result_df['text'] = result_df['text_clean']
    
    # Log completion
    logging.info("Preprocessing complete")
    
    return result_df