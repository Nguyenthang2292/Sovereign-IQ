# Tóm tắt triển khai Agent MemOS Workflow Memory

## 1. Design doc (đã cập nhật Part 1 + Part 2)

**File:** `docs/plans/2026-02-06-agent-memos-workflow-design.md`

- **Phần 1:** Mục tiêu cho Cursor, OpenCode, Claude Code, **Antigravity**; recall/store tự động; giới hạn YAGNI.
- **Phần 2:** Vị trí module, wrapper MemOS, format file context.

---

## 2. Module `modules/agent_memory/`

| File | Chức năng |
|------|-----------|
| `config.py` | `user_id`, `conversation_id`, `get_context_path()` (env + mặc định). |
| `client.py` | Wrapper MemOS: `get_client`, `add_message`, `search_memory` (optional dependency, no-op nếu không có API key / package). |
| `recall.py` | `run_recall()`: 2 query (conventions + workflow) → ghi `.cursor/agent_memory_context.md`. |
| `store.py` | `store_summary(text)`, `commit_summary(repo_root)` cho git hook. |
| `cli.py` | Subcommands `recall` và `store` (store không args = last commit). |
| `__main__.py` | Entry cho `python -m modules.agent_memory`. |
| `README.md` | Hướng dẫn setup và lệnh. |

---

## 3. Cursor rule

**File:** `.cursor/rules/agent_memory.mdc`

- Nội dung: agent đọc `.cursor/agent_memory_context.md` khi bắt đầu session.

---

## 4. Kiểm tra đã chạy

- `python -m modules.agent_memory recall` chạy thành công.
- File `.cursor/agent_memory_context.md` được tạo (khi chưa có `MEMOS_API_KEY` thì 2 section là "(No memories found.)").

---

## 5. Cách dùng tiếp

- Set `MEMOS_API_KEY` (và tùy chọn `MEMOS_USER_ID`, `MEMOS_CONVERSATION_ID`).
- Cài `pip install MemoryOS` nếu cần gọi MemOS thật.
- Khi bắt đầu làm việc: chạy `python -m modules.agent_memory recall` (hoặc task trong IDE).
- Tùy chọn: post-commit hook gọi `python -m modules.agent_memory store` để tự lưu workflow từ commit.
