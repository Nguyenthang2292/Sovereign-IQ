import os
import subprocess
import sys

import boto3
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv("../../../.env")

# Initialize AWS Session
session = boto3.Session()
sts = session.client("sts")
account_id = sts.get_caller_identity()["Account"]
region = session.region_name or "us-east-1"

ecr = session.client("ecr", region_name=region)

repo_name = "xgboost-trainer"
registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"

# 1. Ensure ECR repository exists
try:
    ecr.describe_repositories(repositoryNames=[repo_name])
    print(f"Repository {repo_name} already exists.")
except ecr.exceptions.RepositoryNotFoundException:
    print(f"Creating repository {repo_name}...")
    ecr.create_repository(repositoryName=repo_name)

# 2. Get Docker Login Auth Token
print("Getting ECR authorization token...")
auth = ecr.get_authorization_token()
token = auth["authorizationData"][0]["authorizationToken"]
import base64

decoded = base64.b64decode(token).decode("utf-8")
username, password = decoded.split(":")

# 3. Docker Login
print(f"Logging in to Docker registry {registry}...")
login_cmd = ["docker", "login", "--username", username, "--password-stdin", registry]
proc = subprocess.Popen(login_cmd, stdin=subprocess.PIPE, text=True)
proc.communicate(input=password)
if proc.returncode != 0:
    print("Docker login failed.")
    sys.exit(proc.returncode)

# 4. Docker Build
image_uri = f"{registry}/{repo_name}:latest"
print(f"Building Docker image {repo_name}...")
build_cmd = [
    "docker",
    "build",
    "--provenance=false",
    "-f",
    "modules/xgboost_LTS_serverless/lambda/trainer/Dockerfile",
    "-t",
    f"{repo_name}:latest",
    ".",
]
# Run the build from the project root directory
proc = subprocess.run(build_cmd, cwd="../../..")
if proc.returncode != 0:
    print("Docker build failed.")
    sys.exit(proc.returncode)

# 5. Docker Tag
print(f"Tagging image to {image_uri}...")
proc = subprocess.run(["docker", "tag", f"{repo_name}:latest", image_uri])
if proc.returncode != 0:
    print("Docker tag failed.")
    sys.exit(proc.returncode)

# 6. Docker Push
print(f"Pushing image {image_uri}...")
proc = subprocess.run(["docker", "push", image_uri])
if proc.returncode != 0:
    print("Docker push failed.")
    sys.exit(proc.returncode)

print("Docker build and push completed successfully!")

# 7. Run CloudFormation Deployment Script
print("Starting CloudFormation stack deployment...")
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deploy_trainer_stack.py")
proc = subprocess.run([sys.executable, script_path])
if proc.returncode != 0:
    print("Deployment failed.")
    sys.exit(proc.returncode)

print("Deployment completed successfully!")
