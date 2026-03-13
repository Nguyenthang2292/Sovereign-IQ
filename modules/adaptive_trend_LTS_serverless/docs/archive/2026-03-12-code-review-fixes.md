# Code Review Fixes — adaptive_trend_LTS_serverless

> Review date: 2026-03-12
> Reviewer: Claude Code
> Status: Pending fixes

---

## CRITICAL

### 1. Lambda Function URL without authentication

- **File**: `scripts/fix_lambda_url.py` (lines 42-64)
- **Severity**: CRITICAL — Security
- **Problem**: `AuthType="NONE"` + `Principal="*"` makes Lambda publicly accessible. Anyone can invoke trading endpoint.
- **Fix**:

```python
# BEFORE (INSECURE)
response = lambda_client.create_function_url_config(
    FunctionName=function_name,
    AuthType="NONE",  # <-- No authentication
)
lambda_client.add_permission(
    FunctionName=function_name,
    StatementId="FunctionURLAllowPublicAccess",
    Action="lambda:InvokeFunctionUrl",
    Principal="*",  # <-- Anyone on the internet
    FunctionUrlAuthType="NONE",
)

# AFTER (SECURE)
response = lambda_client.create_function_url_config(
    FunctionName=function_name,
    AuthType="AWS_IAM",  # Require IAM authentication
)
# Remove the add_permission call entirely — IAM auth handles access control
```

- **Alternative**: Delete `fix_lambda_url.py` entirely. The main `lambda_client.py` already uses boto3 IAM-signed invocation, so this script is unnecessary.

---

### 2. Overly broad IAM policy

- **File**: `scripts/deploy_lambda.py` (line 149)
- **Severity**: CRITICAL — Security
- **Problem**: `AmazonSQSFullAccess` grants full access to ALL SQS queues in the account.
- **Fix**:

```python
# BEFORE
"arn:aws:iam::aws:policy/AmazonSQSFullAccess",

# AFTER — Replace managed policy with inline policy:
iam_client.put_role_policy(
    RoleName=role_name,
    PolicyName="ATCSQSSendOnly",
    PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "sqs:SendMessage",
            "Resource": f"arn:aws:sqs:{region}:{account_id}:{queue_name}"
        }]
    })
)
```

---

### 3. Deprecated `np.random.seed()` global state

- **File**: `benchmarks/benchmark_atc_comparison.py` (line 88)
- **Severity**: CRITICAL — Bug risk in parallel execution
- **Problem**: Global random state causes non-deterministic results when tests run in parallel.
- **Fix**:

```python
# BEFORE
np.random.seed(seed)
opens = [base_price]
for i in range(1, num_bars):
    change = np.random.normal(0, volatility)
    opens.append(opens[-1] * (1 + change))
highs = [o * (1 + abs(np.random.normal(0, volatility * 0.5))) for o in opens]
lows = [o * (1 - abs(np.random.normal(0, volatility * 0.5))) for o in opens]

# AFTER
rng = np.random.default_rng(seed)
opens = [base_price]
for i in range(1, num_bars):
    change = rng.normal(0, volatility)
    opens.append(opens[-1] * (1 + change))
highs = [o * (1 + abs(rng.normal(0, volatility * 0.5))) for o in opens]
lows = [o * (1 - abs(rng.normal(0, volatility * 0.5))) for o in opens]
```

---

## IMPORTANT

### 4. Shallow copy of nested config dict

- **Files**:
  - `lambda_client.py` (line 169)
  - `scripts/binance_lambda_demo.py` (line 183)
  - `examples/python_client.py`
- **Severity**: HIGH — Bug
- **Problem**: `.copy()` is shallow — nested `ma_configs` (list of dicts) remains shared. Mutating it corrupts the default for all subsequent calls.
- **Fix**:

```python
# BEFORE
import copy  # Add this import at top of file

config = DEFAULT_ATC_CONFIG.copy()

# AFTER
config = copy.deepcopy(DEFAULT_ATC_CONFIG)
```

Apply this change in all 3 files listed above.

---

### 5. ATCLambdaClient duplicated in 3 files

- **Files**:
  - `lambda_client.py` (original)
  - `scripts/binance_lambda_demo.py` (lines 131-265) — full copy
  - `examples/python_client.py` (lines 22-198) — copy as `ATCServerlessClient`
- **Severity**: HIGH — Maintainability
- **Problem**: Bug fixes to the main client don't propagate to copies.
- **Fix**:

```python
# In scripts/binance_lambda_demo.py — replace the duplicated class with:
from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient, DEFAULT_ATC_CONFIG

# In examples/python_client.py — replace the duplicated class with:
from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient, DEFAULT_ATC_CONFIG
# Rename ATCServerlessClient references to ATCLambdaClient
```

---

### 6. Division by zero in benchmark display

- **File**: `benchmarks/benchmark_atc_comparison.py` (lines 566, 749)
- **Severity**: MEDIUM — Bug
- **Problem**: `matches / total * 100` crashes with `ZeroDivisionError` if both result lists are empty.
- **Fix**:

```python
# BEFORE
match_pct = matches / total * 100

# AFTER
match_pct = (matches / total * 100) if total > 0 else 0.0
```

---

### 7. Missing type annotations

- **File**: `generate_test_data.py` (lines 6, 46, 162)
- **Severity**: MEDIUM — Code quality
- **Fix**:

```python
# BEFORE
def generate_ohlcv(num_bars=200):
def generate_symbol_data():
def generate_test_data(num_symbols=120):

# AFTER
from typing import Any

def generate_ohlcv(num_bars: int = 200) -> dict[str, list[float] | list[int]]:
def generate_symbol_data() -> dict[str, Any]:
def generate_test_data(num_symbols: int = 120) -> dict[str, Any]:
```

---

### 8. XSS in HTML report generation

- **File**: `scripts/benchmark_tracking.py` (line 142)
- **Severity**: MEDIUM — Security hygiene
- **Problem**: Benchmark names interpolated directly into HTML without escaping.
- **Fix**:

```python
# Add import at top
import html

# BEFORE
html_parts.append(f"""        <tr>
    <td>{result['name']}</td>

# AFTER
html_parts.append(f"""        <tr>
    <td>{html.escape(result['name'])}</td>
```

Apply `html.escape()` to all user-derived values in the HTML template.

---

### 9. No unit tests for lambda_client.py

- **Severity**: HIGH — Test coverage
- **Problem**: Core exported module has zero test coverage.
- **Action**: Create `tests/adaptive_trend_LTS_serverless/test_lambda_client.py`
- **Test cases needed**:

```
[x] ATCLambdaClient mock mode invoke — returns valid result
[x] ATCLambdaClient mock mode batch invoke — returns results for all symbols
[x] _poll_sqs_for_batch timeout handling — returns partial results after timeout
[x] Error handling — Lambda invocation error returns error dict
[x] Error handling — malformed SQS message is skipped gracefully
[x] DEFAULT_ATC_CONFIG deepcopy — mutations don't affect original
```

---

### 10. Legacy `typing.Dict/List` imports

- **Files**:
  - `examples/python_client.py` (lines 14-15, 56, 103)
  - `scripts/binance_lambda_demo.py` (lines 30, 83, 102)
- **Severity**: LOW — Code style
- **Fix**:

```python
# BEFORE
from typing import Any, Dict, List

def some_func(data: List[Dict[str, Any]]) -> Dict[str, Any]:

# AFTER
from typing import Any

def some_func(data: list[dict[str, Any]]) -> dict[str, Any]:
```

---

## Checklist

- [x] **CRITICAL #1** — Fix or remove `fix_lambda_url.py`
- [x] **CRITICAL #2** — Scope IAM policy in `deploy_lambda.py`
- [x] **CRITICAL #3** — Replace `np.random.seed()` with `default_rng()`
- [x] **HIGH #4** — `copy.deepcopy()` in 3 files
- [x] **HIGH #5** — Remove duplicated `ATCLambdaClient` in 2 files
- [x] **MEDIUM #6** — Division by zero guard in benchmark
- [x] **MEDIUM #7** — Type annotations in `generate_test_data.py`
- [x] **MEDIUM #8** — `html.escape()` in benchmark HTML report
- [x] **HIGH #9** — Create `test_lambda_client.py`
- [x] **LOW #10** — Replace legacy `typing.Dict/List`
