"""
ATC Serverless Module

This module provides the client for interacting with the ATC implementation deployed on AWS Lambda.
Main component:
    - ATCLambdaClient: Invokes Lambda (via boto3) and reads the direct ScanResult response.
"""

from .lambda_client import DEFAULT_ATC_CONFIG, ATCLambdaClient

__all__ = ["ATCLambdaClient", "DEFAULT_ATC_CONFIG"]
