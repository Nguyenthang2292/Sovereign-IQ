# Migration Tool (SQLite → DynamoDB)

Bộ công cụ này hỗ trợ migrate dữ liệu AutoTrade từ SQLite sang DynamoDB, xác minh integrity sau migrate, và rollback khẩn cấp.

## 1) Dry-run trước khi migrate

```bash
python -m modules.auto_trade.database.migration_tool.sqlite_to_dynamodb \
  --sqlite-path data/auto_trade.db \
  --dynamodb-table AutoTrade \
  --region ap-southeast-1 \
  --dry-run
```

## 2) Chạy migrate thực tế

```bash
python -m modules.auto_trade.database.migration_tool.sqlite_to_dynamodb \
  --sqlite-path data/auto_trade.db \
  --dynamodb-table AutoTrade \
  --region ap-southeast-1 \
  --batch-size 100 \
  --verbose
```

## 3) Verify sau migrate

```bash
python -m modules.auto_trade.database.migration_tool.verify_migration \
  --sqlite-path data/auto_trade.db \
  --dynamodb-table AutoTrade \
  --region ap-southeast-1 \
  --sample-size 10
```

Có thể dùng dry-run cho verify để chỉ xem count trên SQLite:

```bash
python -m modules.auto_trade.database.migration_tool.verify_migration \
  --sqlite-path data/auto_trade.db \
  --dry-run
```

## 4) Rollback khẩn cấp

```bash
python -m modules.auto_trade.database.migration_tool.rollback \
  --dynamodb-table AutoTrade \
  --region ap-southeast-1 \
  --force
```

## Ghi chú
- `sqlite_to_dynamodb` sẽ transform dữ liệu sang single-table schema (pk/sk + GSIs), convert float → Decimal, datetime → ISO-8601.
- `verify_migration` kiểm tra count và spot-check random theo từng entity.
- `rollback` xóa toàn bộ dữ liệu trong table DynamoDB, chỉ dùng khi thật sự cần.
