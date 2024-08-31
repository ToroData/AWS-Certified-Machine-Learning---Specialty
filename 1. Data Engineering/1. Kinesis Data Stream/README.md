# 1. Kinesis Data Stream

## Overview

This project is part of the `1. Data Engineering` directory under the `AWS-Certified-Machine-Learning-Specialty` repository. It demonstrates the integration of a machine learning model training pipeline with AWS Kinesis Data Streams for logging training progress in real-time.

## Video Tutorial

[Watch the video](../../videos/Data%20Engineering/1.%20Kinesis%20Data%20Stream.mp4)

### Key Features

1. **Data Loading and Preprocessing**:
   - Load and preprocess the Iris dataset.
   - Standardize features and split the dataset into training and testing sets.

2. **Model Training**:
   - Train a Random Forest model with hyperparameter tuning using `GridSearchCV`.
   - Train the model over a configurable number of epochs.

3. **AWS Kinesis Logging**:
   - Log the model's accuracy after each epoch to an AWS Kinesis Data Stream.
   - Capture and print the response from AWS Kinesis to the console.

4. **Model Saving**:
   - Save the trained model to a `.pkl` file for future use.

## Configuration

The project configuration is managed in the `src/config.py` file:

```python
CONFIG = {
    'model_path': 'models/random_forest_model.pkl',
    'kinesis_stream_name': 'ML_Model_Training_Kinesis_Log',
    'region_name': 'eu-west-3',
    'epochs': 10
}
```

- model_path: Path to save the trained model.
- kinesis_stream_name: Name of the Kinesis stream for logging training data.
- region_name: AWS region where the Kinesis stream is located.
- epochs: Number of training epochs.

### Project Structure

```plaintext

AWS-Certified-Machine-Learning-Specialty/
└── 1. Data Engineering/
    └── 1. Kinesis Data Stream/
        ├── models/
        │   └── random_forest_model.pkl      # Trained model file
        ├── src/
        │   ├── __init__.py                  # Module initialization file
        │   ├── config.py                    # Configuration settings
        │   ├── data_preprocessing.py        # Data loading and preprocessing logic
        │   ├── kinesis_logging.py           # AWS Kinesis logging logic
        │   ├── model_training.py            # Model training and evaluation logic
        │   ├── utils.py                     # Additional utilities (currently unused)
        ├── main.py                          # Main script to execute the pipeline
        ├── requirements.txt                 # Project dependencies
```

### Dependencies

Ensure the following dependencies are installed before running the project:

```plaintext
boto3
pandas
scikit-learn
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

### Usage

To run the project, make sure all configurations are correctly set in src/config.py. Then, execute the following command from the 1. Kinesis Data Stream directory:

```bash
python main.py
```

The script will:

- Load and preprocess the Iris dataset.
- Train the model over the specified number of epochs.
- Log the training accuracy to AWS Kinesis after each epoch.
- Save the trained model to the specified file path.

### License

This project is licensed under the MIT License. Please refer to the LICENSE file at the root of the repository for more details.
