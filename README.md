# Disaster Tweet AI

A machine learning project for classifying tweets as disaster-related or not using natural language processing techniques.

Put your training data (`train.csv`) in `/data` folder.
Configure your environment variables in `.env` if needed for API integrations.

# Table of content

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Disaster Tweet AI is a machine learning project that uses natural language processing to classify tweets as disaster-related or not. The project includes comprehensive data analysis, preprocessing, and modeling pipelines using Python and scikit-learn.

The project is structured around three main phases:
- **Exploratory Data Analysis (EDA)**: Understanding the dataset structure and characteristics
- **Data Preprocessing**: Cleaning and preparing text data for modeling
- **Modeling & Evaluation**: Building and evaluating classification models

## Installation
You will need Python 3.8+ and Docker installed on your machine to run this project.

You will also need to place your training data file (`train.csv`) in the `/data` folder.

Optionally, create a `.env` file for any API keys or configuration variables you might need.

## Usage
Install the dependencies with the following command:
```bash
pip install -r requirements.txt
```

### Running Jupyter Notebooks
Start Jupyter to explore the analysis notebooks:
```bash
jupyter lab
```

Then navigate to the `src/` folder to access the notebooks:
- `analysis/exploratory_analysis.ipynb` - Data exploration and analysis
- `Preprocessing.ipynb` - Data cleaning and preparation
- `Modeling.ipynb` - Model building and evaluation

### Running the Pipeline
Execute the complete pipeline with:
```bash
python src/pipeline_tweet_classifier.py
```

### Running Tests
Run the test suite with:
```bash
python -m pytest tests/
```

## Project Structure
```
Disaster_Tweet_AI/
├── data/                          # Data directory
│   └── train.csv                 # Training dataset
├── src/                          # Source code
│   ├── analysis/                 # Analysis notebooks
│   │   └── exploratory_analysis.ipynb
│   ├── Preprocessing.ipynb       # Data preprocessing
│   ├── Modeling.ipynb           # Model development
│   └── pipeline_tweet_classifier.py  # Main pipeline
├── tests/                        # Test files
├── docker-compose.yml           # Docker compose configuration
├── Dockerfile                   # Docker configuration
└── requirements.txt             # Python dependencies
```

## Notebooks

### 1. Exploratory Data Analysis
- Data structure analysis
- Missing values and duplicates detection
- Text length statistics
- Class distribution analysis

### 2. Data Preprocessing
- Text cleaning and normalization
- Feature engineering
- Data transformation pipelines

### 3. Modeling & Evaluation
- Model training with various algorithms
- Performance evaluation
- Hyperparameter tuning
- Model comparison

## Docker
You can also run the project using Docker:

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up
```

## Contributing
If you want to contribute to this project you can fork this repository and make a pull request with your changes.
Anyone is welcome to contribute to this project.

### Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure notebooks run without errors

## License
This project is under the MIT license.