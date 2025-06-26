"""
Pytest configuration and fixtures for disaster tweet classification tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_texts():
    """Sample texts for testing preprocessing functions."""
    return [
        "EARTHQUAKE hits California! Buildings collapse https://news.com #disaster @emergency",
        "Beautiful sunny day at the beach :) #vacation",
        "WILDFIRE spreading rapidly!!! Evacuations ordered NOW! #emergency",
        "Just had a great coffee this morning 123",
        "",  # Empty string
        None,  # None value
        "RT @user: Fire emergency!!! Call 911 #fire",
        "flooding in the streets water everywhere #flood #emergency",
        "Normal tweet about my day at work",
        "TSUNAMI WARNING: waves approaching coast #tsunami #warning"
    ]


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing preprocessing and modeling functions."""
    data = {
        'text': [
            "EARTHQUAKE hits California! Buildings collapse https://news.com #disaster",
            "Beautiful sunny day at the beach :) #vacation", 
            "WILDFIRE spreading rapidly!!! Evacuations ordered NOW!",
            "Just had a great coffee this morning",
            "RT @user: Fire emergency!!! Call 911 #fire",
            "flooding in the streets water everywhere #flood",
            "Normal tweet about my day at work",
            "TSUNAMI WARNING: waves approaching coast #tsunami",
            "",  # Empty string
            None  # None value
        ],
        'target': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def clean_sample_dataframe():
    """Sample dataframe with clean data for modeling tests."""
    data = {
        'text': [
            "earthquake hits california buildings collapse",
            "beautiful sunny day beach",
            "wildfire spreading rapidly evacuations ordered",
            "great coffee morning",
            "fire emergency call",
            "flooding streets water everywhere",
            "normal tweet day work",
            "tsunami warning waves approaching coast"
        ],
        'target': [1, 0, 1, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_model_file():
    """Temporary file for saving/loading models."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        yield tmp.name
    # Clean up
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def temp_csv_file():
    """Temporary CSV file for testing data loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        # Write sample CSV data
        tmp.write("text,target\n")
        tmp.write("earthquake california,1\n")
        tmp.write("sunny day beach,0\n")
        tmp.write("wildfire evacuation,1\n")
        tmp.write("coffee morning,0\n")
        tmp.flush()
        yield tmp.name
    # Clean up
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
