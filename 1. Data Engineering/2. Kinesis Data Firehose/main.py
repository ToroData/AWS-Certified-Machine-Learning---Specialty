"""
Main script for training a machine learning model and logging progress to AWS Kinesis.

This script performs the following tasks:
1. Loads and preprocesses data from a specified path.
2. Trains a machine learning model for a number of epochs.
3. Evaluates the model at each epoch and logs the results to an AWS Kinesis Data Stream.
4. Saves the trained model to a specified file path.

Dependencies:
- src.data_preprocessing: Contains functions for loading and preprocessing data.
- src.model_training: Contains functions for training, evaluating, and saving the model.
- src.kinesis_logging: Contains function for logging data to AWS Kinesis.
- src.config: Contains configuration settings for file paths and hyperparameters.

Configuration:
- `data_path`: Path to the CSV file containing the data.
- `epochs`: Number of training epochs.
- `kinesis_stream_name`: Name of the AWS Kinesis Data Stream for logging.
- `model_path`: Path to save the trained model.

Usage:
Ensure that the configuration file and necessary dependencies are properly set up before running this script.

Example:
    $ python main_script.py

Author: Ricard Santiago Raigada García
Date: 31-08-2024
"""
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model, save_model
from src.kinesis_logging import send_log_to_kinesis
from src.config import CONFIG
import warnings
warnings.filterwarnings("ignore", message="A NumPy version >=")


def main():
    """
    Main function to orchestrate the model training pipeline and logging.

    This function performs the following steps:
    1. Load and Prepare Data:
       - Loads the data from the file path specified in the configuration.
       - Preprocesses the data to prepare it for model training and evaluation.

    2. Train the Model and Log Results:
       - Trains the model for a number of epochs as specified in the configuration.
       - Evaluates the model's accuracy at each epoch.
       - Sends a log entry with the epoch number and accuracy to an AWS Kinesis Data Stream.
       - Prints the response from Kinesis to the console.

    3. Save the Trained Model:
       - Saves the trained model to the file path specified in the configuration.

    Configuration:
    - `CONFIG['data_path']`: Path to the CSV file containing the data.
    - `CONFIG['epochs']`: Number of training epochs.
    - `CONFIG['kinesis_stream_name']`: Name of the AWS Kinesis Data Stream for logging.
    - `CONFIG['model_path']`: Path to save the trained model.

    Dependencies:
    - `src.data_preprocessing`: For functions to load and preprocess data.
    - `src.model_training`: For functions to train, evaluate, and save the model.
    - `src.kinesis_logging`: For logging data to AWS Kinesis.
    - `src.config`: For configuration settings.

    Example Usage:
        If this script is run directly, it will execute the `main` function.
        Ensure that all necessary configurations are set in the `CONFIG` and required modules are installed.

    """
    # 1. Load and Prepare Data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # 2. Train the Model and Log Results
    for epoch in range(1, CONFIG['epochs'] + 1):
        model = train_model(X_train, y_train)

        accuracy = evaluate_model(model, X_test, y_test)

        # Log a Kinesis
        response = send_log_to_kinesis(
            CONFIG['kinesis_stream_name'], CONFIG['region_name'], epoch, accuracy)
        print(f"Kinesis response: {response}")

    # 3. Save the Trained Model
    save_model(model, CONFIG['model_path'])


if __name__ == "__main__":
    main()
