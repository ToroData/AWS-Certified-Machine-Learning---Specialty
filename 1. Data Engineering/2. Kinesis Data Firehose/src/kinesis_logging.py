"""
Module to send logs to an AWS Kinesis Data Stream.

Author: Ricard Santiago Raigada García
Date:31-08-2024
"""
import boto3
import json
from datetime import datetime


def send_log_to_kinesis(stream_name, region_name, epoch, accuracy):
    """
    Send log data to an AWS Kinesis Data Stream

    Args:
        stream_name (str): The name of the Kinesis data stream
        epoch (int): The current training epoch
        accuracy (float): The accuracy of the model at the current epoch

    Returns:
        dict: Response from the Kinesis `put_record` operation
    """
    kinesis = boto3.client('kinesis', region_name='eu-west-3')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        'timestamp': current_time,
        'epoch': epoch,
        'accuracy': accuracy,
        'message': f"Training at epoch {epoch} - Accuracy: {accuracy}"
    }

    response = kinesis.put_record(
        StreamName=stream_name,
        Data=json.dumps(payload),
        PartitionKey="partition"
    )
    return response
