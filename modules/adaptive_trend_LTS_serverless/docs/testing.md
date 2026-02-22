# Testing Guide

This document explains how to set up and run tests for the ATC Serverless module.

## LocalStack Testing

LocalStack provides a local AWS cloud environment for testing Lambda functions, SQS, and other AWS services.

### Prerequisites

- Docker Desktop
- AWS CLI
- SAM CLI

### Setup

1. Start LocalStack:
   ```bash
   docker-compose up -d
   ```

2. Wait for LocalStack to initialize (approximately 30 seconds).

3. Verify LocalStack is running:
   ```bash
   aws --endpoint-url=http://localhost:4566 lambda list-functions
   ```

### Running Tests

Run the integration tests:
```bash
cargo test --test lambda_integration_tests
```

### Cleaning Up

Stop LocalStack when done:
```bash
docker-compose down
```

## SAM Local Testing

AWS SAM (Serverless Application Model) provides an alternative local testing environment.

### Prerequisites

- SAM CLI
- Docker (for containerized functions)

### Setup

1. Build the Lambda function:
   ```bash
   sam build
   ```

2. Start the local API:
   ```bash
   sam local start-api
   ```

### Running Tests

Run the integration tests with SAM:
```bash
cargo test --test lambda_integration_tests
```

### Cleaning Up

Stop the local API server:
```bash
Ctrl+C
```

## Test Suite

The integration tests cover:

- LocalStack setup and connectivity
- Lambda function deployment
- SQS message sending
- CloudWatch logging
- Error scenario handling

## Troubleshooting

### Common Issues

- **LocalStack not starting**: Check Docker is running and ports 4566, 4571 are available
- **SAM build failures**: Ensure Rust toolchain is properly configured
- **Test failures**: Check LocalStack/SAM is running and accessible

### Debugging

Enable detailed logging:
```bash
RUST_LOG=debug cargo test --test lambda_integration_tests
```

## CI/CD Integration

The tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set up Docker
        uses: docker/setup-docker@v3
      
      - name: Start LocalStack
        run: docker-compose up -d
      
      - name: Run integration tests
        run: cargo test --test lambda_integration_tests
      
      - name: Stop LocalStack
        run: docker-compose down
```

## Additional Resources

- [LocalStack Documentation](https://docs.localstack.cloud/)
- [SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html)