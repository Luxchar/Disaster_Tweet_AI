"""
Main module for disaster tweet classification.
This module provides a high-level interface for loading models and making predictions.
"""

import os
import joblib
import logging
import pandas as pd
from typing import Union, List
from lib.preprocessing import clean_text, preprocess_dataframe
from lib.modeling import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DisasterTweetClassifier:
    """
    High-level interface for disaster tweet classification.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            if model_path.endswith('.pkl'):
                # Load model using joblib
                self.model = joblib.load(model_path)
                logger.info("Loaded model from %s", model_path)
            else:
                raise ValueError(f"Unsupported model file format: {model_path}")
            
            self.model_path = model_path
            
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model_path, e)
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        return clean_text(text)
    
    def predict(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Predict whether text(s) describe disaster events.
        
        Args:
            text (str or List[str]): Text or list of texts to classify
            
        Returns:
            int or List[int]: Prediction(s) - 1 for disaster, 0 for non-disaster
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        # clean text with preprocessing
        preprocessed = self.preprocess_text(text)

        if hasattr(self.model, 'predict'):
            print("Preprocessed text:", preprocessed)
            prediction = self.model.predict([preprocessed])[0]
            print("Prediction:", prediction)
            return int(prediction)

    def train_new_model(self, data_path: str, text_column: str = 'text', 
                       target_column: str = 'target', save_path: str = None):
        """
        Train a new model on provided data.
        
        Args:
            data_path (str): Path to CSV file with training data
            text_column (str): Name of the text column
            target_column (str): Name of the target column
            save_path (str, optional): Path to save the trained model
        """
        logger.info("Loading training data from %s", data_path)
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Ensure target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Preprocess data
        df_processed = preprocess_dataframe(df, text_column)
        logger.info(f"After preprocessing: {len(df_processed)} samples remaining")
        
        # Ensure we still have the target column after preprocessing
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' lost during preprocessing")
        
        # Train model
        logger.info("Training new model...")
        
        # Prepare training data with proper column names
        df_train = df_processed[['text_clean', target_column]].copy()
        df_train = df_train.rename(columns={'text_clean': 'text'})
        
        logger.info(f"Training data shape: {df_train.shape}")
        logger.info(f"Text column samples: {len(df_train['text'])}")
        logger.info(f"Target column samples: {len(df_train[target_column])}")
        
        pipeline, train_results = train_model(df_train)
        self.model = pipeline
        
        logger.info("Training completed: %s", train_results)
        
        # Save model if path provided
        if save_path:
            joblib.dump(self.model, save_path)
            self.model_path = save_path


def display_menu():
    """Display the main menu options."""
    print("\n" + "="*50)
    print("   Disaster Tweet Classification CLI")
    print("="*50)
    print("1. Train a new model")
    print("2. Use existing model for predictions")
    print("3. Exit")
    print("="*50)


def train_model_interface():
    """Interface for training a new model."""
    print("\n--- Training New Model ---")
    
    try:
        classifier = DisasterTweetClassifier()
        logger.info("Starting model training...")
        
        classifier.train_new_model(
            data_path='../data/train.csv',
            text_column='text',
            target_column='target',
            save_path='../models/disaster_tweet_model_main.pkl'
        )
        
        print("\n✓ Model trained successfully and saved!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("Training failed. Please check the logs for details.")


def prediction_interface():
    """Interface for making predictions with existing model."""
    print("\n--- Making Predictions ---")
    
    # Use the main model path
    model_path = '../models/disaster_tweet_model.pkl'
    
    try:
        # Load model
        classifier = DisasterTweetClassifier(model_path=model_path)
        logger.info("Loading existing model...")
        classifier.load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Prediction loop
        while True:
            print("\n--- Enter text to classify ---")
            print("(Type 'back' to return to main menu)")
            
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'back':
                break
            elif not text:
                print("Please enter some text.")
                continue
            
            try:
                prediction = classifier.predict(text)
                result = "DISASTER" if prediction == 1 else "NON-DISASTER"
                confidence = "🔴" if prediction == 1 else "🟢"
                
                print(f"\n{confidence} Prediction: {result}")
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                print("Failed to make prediction. Please try again.")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("Failed to load model. Please check the model file.")


def main():
    """
    Main CLI interface for disaster tweet classification.
    """
    print("Welcome to Disaster Tweet Classification!")
    
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                train_model_interface()
            
            elif choice == '2':
                prediction_interface()
            
            elif choice == '3':
                print("\nGoodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print("An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    main()