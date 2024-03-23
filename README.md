# Classify API Requests

This machine-learning model classifies textual user requests into two categories: actionable requests requiring an API call and purely informative requests. It uses Natural Language Processing (NLP) to understand and categorize requests. It is meant to be a more scalable and resource-efficient alternative to LLM-based solutions.

## Overview

The project uses a Logistic Regression model trained with Scikit-learn, using TF-IDF vectorization for feature extraction from textual data. The current `data.json` file contains a broad range of questions.

### Dataset

The dataset is a collection of textual user requests labeled as requiring an API call (`1`) or not (`0`). It should be stored in a `data.json` file with the following format:

```json
[
    {"text": "Example request 1", "label": 0},
    {"text": "Example request 2", "label": 1},
    ...
]
```

### Training the Model

The `train_model.py` script trains the Logistic Regression model using the provided dataset, evaluates its performance, and saves the trained model and vectorizer for later use.

### Making Predictions

The `predict.py` script allows for classifying new user requests using the trained model. It prompts the user to enter a request and outputs the classification.

## Technologies and Tools

- **Language**: Python
- **Machine Learning Framework**: Scikit-Learn
- **Data Handling**: JSON
- **Model Serialization**: joblib


## Setup and Installation

Ensure you have Python installed on your system.

### 1. Clone the Repository
Clone the project repository and navigate into the project directory.

```bash
git clone https://github.com/aarush-kukreja/classify-api-request
cd classify-api-request
```

### 2. Install Dependencies

You have two options for installing dependencies: directly or using a virtual environment to isolate them. Choose the method that best suits your workflow.

#### Option A: Without Using a Virtual Environment
Directly install the project dependencies by running:

```bash
pip install scikit-learn joblib
```

#### Option B: Using a Virtual Environment
To avoid conflicts with other projects by isolating dependencies, follow these steps to use a virtual environment:

- **For Linux/macOS:**

    Create the virtual environment:
    ```bash
    python3 -m venv myenv
    ```
    
    Activate the virtual environment:
    ```bash
    source myenv/bin/activate
    ```
    
    Install the dependencies:
    ```bash
    pip install scikit-learn joblib
    ```

     To deactivate the virtual environment when you're done, run:
    ```bash
    deactivate
    ```
    
- **For Windows:**

    Create the virtual environment:
    ```bash
    python -m venv myenv
    ```
    
    Activate the virtual environment:
    ```bash
    myenv\Scripts\activate
    ```
    
    Install the dependencies:
    ```bash
    pip install scikit-learn joblib
    ```

    To deactivate the virtual environment when you're done, run:
    ```bash
    deactivate
    ```
    
### 3. **Prepare Your Dataset**

Use the `data.json` file in the project root with your dataset following the structure mentioned in the dataset section.

### 4. **Train the Model**
```bash
python train_model.py
```

### 5. **Run Predictions**
```bash
python predict.py
```
