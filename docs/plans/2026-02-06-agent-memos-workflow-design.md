# Agent MemOS Workflow Memory — Design

**Date:** 2026-02-06  
**Goal:** Module dùng MemOS để agent (Cursor, OpenCode, Claude Code, **Antigravity**) nhớ (1) lịch sử công việc và (2) quy ước dự án; agent chỉ đọc context đã chuẩn bị sẵn (tự động).

---

## 1. Mục tiêu và phạm vi (Part 1)

**Mục tiêu:** Hỗ trợ các agent **Cursor, OpenCode, Claude Code, Antigravity** nhớ workflow và quy ước repo bằng MemOS. Agent không gọi API trực tiếp — chỉ đọc một file context được cập nhật tự động.

**Phạm vi:**
- **Recall:** Khi bắt đầu session, script chạy (user hoặc task Cursor/VS Code) → gọi MemOS `search_memory` → ghi kết quả vào file cố định (ví dụ `.cursor/agent_memory_context.md`). Rule Cursor/IDE: "Đọc file này khi bắt đầu để biết quy ước và workflow gần đây."
- **Store:** Tự động qua **git hook** (post-commit): mỗi commit → script tóm tắt (message + diff ngắn) → `add_message` lên MemOS. Thêm **CLI** để ghi thủ công: `python -m modules.agent_memory store "tóm tắt workflow"`.

**Giới hạn (YAGNI):** Không đọc lịch sử chat từ IDE; không MCP/server trong bước đầu. Chỉ: wrapper MemOS + script recall + hook/CLI store + một file context + rule đọc file.

**Định danh MemOS:** Một `user_id` cố định theo repo (ví dụ `crypto_probability_repo`), `conversation_id` theo ngày hoặc session (ví dụ `2026-02-06`) để nhóm workflow. Query recall: "project conventions" + "recent workflow" để lấy cả quy ước lẫn lịch sử.

---

## 2. Vị trí module, wrapper MemOS, format file context (Part 2)

### 2.1 Vị trí trong repo

- **Module:** `modules/agent_memory/` (cùng cấp với `modules/auto_trade`, `modules/common`, ...).
- **File context (output recall):** `.cursor/agent_memory_context.md` — trong `.cursor/` để agent dễ thấy, có thể thêm vào rule "đọc file này khi bắt đầu session".
- **Cấu hình:** API key và tùy chọn (user_id, conversation_id strategy) đọc từ biến môi trường hoặc file optional (ví dụ `.env` hoặc `config/agent_memory.yaml`). Không commit API key; `.cursor/agent_memory_context.md` có thể gitignore nếu muốn (mỗi máy recall riêng).

**Cấu trúc thư mục:**

```
modules/agent_memory/
  __init__.py
  client.py       # Wrapper MemOS (add_message, search_memory)
  config.py       # user_id, conversation_id, context path, env
  recall.py       # Script: search → ghi .cursor/agent_memory_context.md
  store.py        # add_message từ messages hoặc summary (git hook / CLI)
  cli.py          # Entry: recall, store (python -m modules.agent_memory recall|store [...])
README.md
```

- **Git hook:** Gợi ý đặt script gọi `store.commit_summary(repo_root)` trong `post-commit`; script đọc last commit (message + diff) → format messages → `client.add_message(...)`.

### 2.2 Wrapper MemOS (client.py)

- **Dependency:** Package `MemoryOS` (pip), dùng `from memos.api.client import MemOSClient` như tài liệu MemOS. Nếu không cài được thì import optional: khi gọi wrapper mà không có MemOS thì no-op hoặc log warning, không làm crash repo.
- **Interface:**
  - `get_client(api_key: str | None = None) -> MemOSClient | None`: khởi tạo client (api_key từ tham số hoặc env `MEMOS_API_KEY`). Trả về `None` nếu thiếu key hoặc import lỗi.
  - `add_message(messages: list[dict], user_id: str, conversation_id: str, *, api_key: str | None = None) -> dict | None`: gọi `client.add_message(...)`, trả về response hoặc None khi lỗi.
  - `search_memory(query: str, user_id: str, conversation_id: str | None = None, *, api_key: str | None = None) -> list | dict | None`: gọi `client.search_memory(...)` (hoặc method tương đương trong MemOS API), trả về kết quả tìm kiếm hoặc None.
- **Id strategy:** `user_id` và `conversation_id` lấy từ `config` (config lấy từ env hoặc file); mặc định `user_id = "crypto_probability_repo"`, `conversation_id = date.today().isoformat()` hoặc env `MEMOS_CONVERSATION_ID`.

### 2.3 Format file context (output recall)

- **File:** `.cursor/agent_memory_context.md`
- **Nội dung (Markdown):**
  - Phần cố định ở đầu: "Context below is from MemOS for agents (Cursor, OpenCode, Claude Code, Antigravity). Use for project conventions and recent workflow."
  - Một section **Project conventions / Quy ước dự án:** nội dung là kết quả search với query kiểu "project conventions structure coding style" (hoặc query cấu hình được).
  - Một section **Recent workflow / Workflow gần đây:** nội dung là kết quả search với query kiểu "recent workflow tasks done" (hoặc query cấu hình được).
  - Nếu MemOS trả về dạng list/dict, script sẽ flatten thành text (ví dụ mỗi memory một bullet) rồi ghi vào section tương ứng. Nếu không có kết quả → section ghi "(No memories found.)"
- **Encoding:** UTF-8. Ghi đè file mỗi lần recall.

**Ví dụ nội dung:**

```markdown
# Agent memory context (MemOS)

Context below is for agents (Cursor, OpenCode, Claude Code, Antigravity). Use for project conventions and recent workflow.

## Project conventions / Quy ước dự án

- Use pytest for tests; place tests under tests/ or module-specific tests/.
- modules/auto_trade: GUI in gui/, execution in execution/, database in database/.

## Recent workflow / Workflow gần đây

- 2026-02-06: Implemented negative breakeven; added trailing stop step index.
- 2026-02-05: Fresh signal auto-trade design and pipeline signals DB.
```

Recall script sẽ gọi `search_memory` hai lần (hoặc một lần với query tổng hợp) rồi merge kết quả vào hai section trên.

---

## 3. Bước tiếp theo (sau khi validate)

- Implement `modules/agent_memory/`: client, config, recall, store, cli.
- Thêm Cursor rule: "Khi bắt đầu session, đọc `.cursor/agent_memory_context.md` để lấy quy ước và workflow gần đây."
- Gợi ý task/script "Refresh agent memory" (chạy `python -m modules.agent_memory recall`) khi mở project hoặc khi cần.
- Tùy chọn: post-commit hook gọi store từ last commit.
