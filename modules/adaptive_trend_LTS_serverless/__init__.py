"""
ATC Serverless Module

This module provides the client for interacting with the ATC implementation deployed on AWS Lambda.
Main component:
    - ATCLambdaClient: Invokes Lambda (via boto3) and polls SQS for results.
"""

from .lambda_client import DEFAULT_ATC_CONFIG, ATCLambdaClient

__all__ = ["ATCLambdaClient", "DEFAULT_ATC_CONFIG"]
