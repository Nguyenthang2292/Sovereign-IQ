# Cutover Runbook: SQLite to DynamoDB

> **Date**: 2026-02-20  
> **Status**: Ready for Execution

---

## Pre-Cutover Checklist

- [ ] DynamoDB table created and verified (run `python infrastructure/dynamodb/create_table.py --env prod`)
- [ ] IAM permissions tested locally with DynamoDB Local
- [ ] Migration tool dry-run completed: `python -m modules.auto_trade.database.migration_tool.sqlite_to_dynamodb --dry-run`
- [ ] Verification script dry-run: `python -m modules.auto_trade.database.migration_tool.verify_migration --dry-run`
- [ ] Lambda environment variables updated (`DB_BACKEND=dynamodb`, `DYNAMODB_TABLE_NAME=AutoTrade`)
- [ ] Rollback plan reviewed and ready
- [ ] Database backup taken (SQLite): `python -m modules.auto_trade.database.backup`

---

## Cutover Steps

### Step 1: Pause Trading Bot

```bash
# Set system state to disable trading
DB_BACKEND=sqlite python -c "
from modules.auto_trade.database.repository.context import RepositoryContext
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/auto_trade.db')
Session = sessionmaker(bind=engine)
session = Session()

ctx = RepositoryContext.for_sqlite(session)
ctx.system_state.set_system_state('trading_enabled', 'false', 'boolean', 'system')
print('Trading disabled')
session.close()
"
```

### Step 2: Run Final Migration

```bash
# Migrate data from SQLite to DynamoDB
python -m modules.auto_trade.database.migration_tool.sqlite_to_dynamodb \
    --sqlite-path data/auto_trade.db \
    --dynamodb-table AutoTrade \
    --region ap-southeast-1 \
    --verbose
```

Expected output:

```text
Connecting to SQLite: data/auto_trade.db
Connecting to DynamoDB: ap-southeast-1/AutoTrade
Migrating Signals...
  Found X records
  Written X/X records
Migrating Orders...
  ...

=== Migration Complete ===
Total records migrated: X
```

### Step 3: Verify Migration

```bash
# Verify data integrity
python -m modules.auto_trade.database.migration_tool.verify_migration \
    --sqlite-path data/auto_trade.db \
    --dynamodb-table AutoTrade \
    --region ap-southeast-1
```

Expected output:

```text
Verifying Signals...
  Count: ✓ SQLite=X, DynamoDB=X
  Sample: ✓

Verifying Orders...
  ...

==================================================
✓ VERIFICATION PASSED
==================================================
```

### Step 4: Deploy Lambda with DynamoDB Backend

```bash
# Update .env with DynamoDB settings
export DB_BACKEND=dynamodb
export DYNAMODB_TABLE_NAME=AutoTrade
export AWS_REGION=ap-southeast-1

# Deploy Lambda (your deployment command)
python deploy_lambda.py --env prod
```

### Step 5: Smoke Test

```bash
# Test key operations with DynamoDB
DB_BACKEND=dynamodb python -c "
from modules.auto_trade.database.repository.context import RepositoryContext

ctx = RepositoryContext.from_env()

# Test get_open_positions
orders = ctx.orders.get_open_positions()
print(f'Open positions: {len(orders)}')

# Test get_recent_signals
signals = ctx.signals.get_recent_signals(limit=5)
print(f'Recent signals: {len(signals)}')

# Test system state
enabled = ctx.system_state.get_system_state('system.trading_enabled')
print(f'Trading enabled: {enabled}')

print('Smoke test PASSED')
"
```

### Step 6: Resume Trading Bot

```bash
# Re-enable trading
DB_BACKEND=dynamodb python -c "
from modules.auto_trade.database.repository.context import RepositoryContext

ctx = RepositoryContext.from_env()
ctx.system_state.set_system_state('trading_enabled', 'true', 'boolean', 'system')
print('Trading re-enabled')
"
```

### Step 7: Monitor

- Monitor CloudWatch metrics for 30 minutes
- Watch for any errors in Lambda logs
- Check DynamoDB consumed capacity

---

## Rollback Procedure

If issues occur during cutover:

### Option A: Quick Rollback (Revert Lambda)

```bash
# Revert Lambda environment variables
export DB_BACKEND=sqlite

# Redeploy Lambda
python deploy_lambda.py --env prod

# Re-enable trading on SQLite
python -c "
from modules.auto_trade.database.repository.context import RepositoryContext
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/auto_trade.db')
Session = sessionmaker(bind=engine)
session = Session()

ctx = RepositoryContext.for_sqlite(session)
ctx.system_state.set_system_state('trading_enabled', 'true', 'boolean', 'system')
session.close()
print('Rolled back to SQLite')
"
```

### Option B: Emergency Rollback (Clear DynamoDB)

```bash
# ONLY USE IF MAJOR ISSUES - This deletes all DynamoDB data!
python -m modules.auto_trade.database.migration_tool.rollback \
    --dynamodb-table AutoTrade \
    --region ap-southeast-1 \
    --force

# Then follow Option A to revert to SQLite
```

---

## Post-Cutover Tasks

- [ ] Remove old SQLite backup if satisfied with migration
- [ ] Update monitoring dashboards to track DynamoDB metrics
- [ ] Document any issues encountered
- [ ] Schedule 1-week follow-up to verify system stability

---

## Contact Information

- **Primary**: [Your Name]
- **Secondary**: [Backup Contact]
- **Escalation**: [On-Call Team]
