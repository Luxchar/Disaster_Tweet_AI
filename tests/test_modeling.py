"""
Tests for modeling.py functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Add src directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lib.modeling import train_model


class TestTrainModel:
    """Test cases for the train_model function."""
    
    def test_train_model_basic(self, clean_sample_dataframe):
        """Test basic model training functionality."""
        pipeline, results = train_model(clean_sample_dataframe)
        
        # Check that pipeline is returned
        assert isinstance(pipeline, Pipeline)
        
        # Check that results dictionary is returned
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
    
    def test_train_model_pipeline_components(self, clean_sample_dataframe):
        """Test that pipeline has correct components."""
        pipeline, _ = train_model(clean_sample_dataframe)
        
        # Check pipeline steps
        step_names = [name for name, _ in pipeline.steps]
        assert 'tfidf' in step_names
        assert 'clf' in step_names
        
        # Check component types
        tfidf = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['clf']
        
        assert isinstance(tfidf, TfidfVectorizer)
        assert isinstance(clf, LogisticRegression)
    
    def test_train_model_accuracy_range(self, clean_sample_dataframe):
        """Test that accuracy is within reasonable range."""
        _, results = train_model(clean_sample_dataframe)
        
        accuracy = results['accuracy']
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0
    
    def test_train_model_classification_report(self, clean_sample_dataframe):
        """Test that classification report is generated."""
        _, results = train_model(clean_sample_dataframe)
        
        report = results['classification_report']
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
    
    def test_train_model_confusion_matrix(self, clean_sample_dataframe):
        """Test that confusion matrix is generated."""
        _, results = train_model(clean_sample_dataframe)
        
        cm = results['confusion_matrix']
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)  # Binary classification
        assert cm.dtype in [np.int32, np.int64]
    
    def test_train_model_prediction_capability(self, clean_sample_dataframe):
        """Test that trained model can make predictions."""
        pipeline, _ = train_model(clean_sample_dataframe)
        
        # Test single prediction
        test_text = "earthquake california disaster"
        prediction = pipeline.predict([test_text])
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
        
        # Test multiple predictions
        test_texts = ["earthquake disaster", "sunny beach day"]
        predictions = pipeline.predict(test_texts)
        
        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_train_model_prediction_probabilities(self, clean_sample_dataframe):
        """Test that trained model can predict probabilities."""
        pipeline, _ = train_model(clean_sample_dataframe)
        
        test_text = "earthquake california disaster"
        probabilities = pipeline.predict_proba([test_text])
        
        assert probabilities.shape == (1, 2)  # One sample, two classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert all(0.0 <= prob <= 1.0 for prob in probabilities.flatten())
    
    def test_train_model_custom_test_size(self, clean_sample_dataframe):
        """Test model training with custom test size."""
        pipeline, results = train_model(clean_sample_dataframe, test_size=0.3)
        
        # Should still work with different test size
        assert isinstance(pipeline, Pipeline)
        assert 'accuracy' in results
        assert 0.0 <= results['accuracy'] <= 1.0
    
    def test_train_model_required_columns(self):
        """Test that model training requires correct columns."""
        # DataFrame without required columns
        df_missing_text = pd.DataFrame({
            'content': ['test message'],
            'target': [1]
        })
        
        with pytest.raises(KeyError):
            train_model(df_missing_text)
        
        df_missing_target = pd.DataFrame({
            'text': ['test message'],
            'label': [1]
        })
        
        with pytest.raises(KeyError):
            train_model(df_missing_target)
    
    def test_train_model_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame(columns=['text', 'target'])
        
        with pytest.raises(ValueError):
            train_model(df)
    
    def test_train_model_reproducibility(self, clean_sample_dataframe):
        """Test that model training is reproducible with same random state."""
        pipeline1, results1 = train_model(clean_sample_dataframe)
        pipeline2, results2 = train_model(clean_sample_dataframe)
        
        # Due to fixed random_state=42, results should be identical
        assert results1['accuracy'] == results2['accuracy']
        
        # Test predictions should be the same
        test_text = "earthquake california disaster"
        pred1 = pipeline1.predict([test_text])
        pred2 = pipeline2.predict([test_text])
        assert pred1[0] == pred2[0]
