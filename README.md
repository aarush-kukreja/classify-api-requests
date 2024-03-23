# Classify API Project

This is a machine-learning model that classifies textual user requests into two categories: actionable requests requiring an API call and purely informative requests. It uses Natural Language Processing (NLP) to understand and categorize requests. It is meant to be a more scalable and resource-efficient alternative to large language model (LLM)-based solutions.

## Overview

The project uses a Logistic Regression model trained with Scikit-learn, using TF-IDF vectorization for feature extraction from textual data. The current `data.json` file contains a broad range of questions.

## Technologies and Tools

- **Language**: Python
- **Machine Learning Framework**: Scikit-Learn
- **NLP Tools**: NLTK
- **Data Handling**: JSON
- **Model Serialization**: joblib

## Setup and Installation

Ensure you have Python installed on your system. This project was developed using Python 3.8. It is recommended to use a virtual environment for project dependencies.

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install Dependencies (use a virtual environment as needed)**
    ```bash
    pip install scikit-learn nltk joblib pandas matplotlib seaborn
    ```

3. **Prepare Your Dataset**
   - Use the `data.json` file in the project root with your dataset following the structure mentioned in the dataset section.

4. **Train the Model**
    ```bash
    python train_model.py
    ```

5. **Run Predictions**
    ```bash
    python predict.py
    ```

## Dataset

The dataset is a collection of textual user requests labeled as either requiring an API call (`1`) or not (`0`). It should be stored in a `data.json` file with the following format:

```json
[
    {"text": "Example request 1", "label": 0},
    {"text": "Example request 2", "label": 1},
    ...
]
```

## Training the Model

The `train_model.py` script trains the Logistic Regression model using the provided dataset, evaluates its performance, and saves the trained model and vectorizer for later use.

## Making Predictions

The `predict.py` script allows for classifying new user requests using the trained model. It prompts the user to enter a request and outputs the classification.
