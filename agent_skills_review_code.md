# Agent Skills — Review & Audit Code

## 1. Kiểm tra và đánh giá code tổng thể (Review & Audit)

| Skill | Mô tả |
|---|---|
| `code-reviewer` | Chuyên gia review code xuất sắc. Phù hợp nhất khi bạn vừa viết xong một đoạn code và muốn ai đó "đọc lại" để xem logic đã ổn chưa, cách thiết kế có tối ưu không. |
| `production-code-audit` | Rà soát (scan) theo từng dòng code cực kỳ sâu sắc, phân tích kiến trúc, pattern, và logic tổng thể — xem đã đủ tiêu chuẩn để đưa vào production hay chưa. |
| `vibe-code-auditor` | Dành cho code được tạo ra bởi AI hoặc tạo ra rất nhanh. Chuyên đánh giá các điểm có thể gây lỗi cấu trúc, dễ vỡ (fragile) hoặc có rủi ro tiềm ẩn về mặt logic khi chạy thực tế. |
| `comprehensive-review-full-review` | Đánh giá toàn diện trên toàn bộ thay đổi. Lý tưởng khi bạn đang review một luồng xử lý hoặc một tính năng lớn của module. |
| `code-review-checklist` | Checklist review thực chiến để tránh bỏ sót bug logic, bảo mật, hiệu năng, và maintainability. |
| `code-review-excellence` | Nâng chất lượng phản biện code review, tập trung vào rủi ro thực tế thay vì góp ý hình thức. |
| `code-review-ai-ai-review` | Review code có hỗ trợ AI theo hướng tìm lỗi nhanh và phân loại mức độ nghiêm trọng. |
| `codex-review` | Review chuyên nghiệp, phù hợp khi cần đánh giá patch hoặc thay đổi lớn trước khi merge. |
| `architect-review` | Review ở mức kiến trúc: boundaries, coupling, trade-off, và khả năng mở rộng hệ thống. |
| `differential-review` | Soi thay đổi theo diff để phát hiện regression và rủi ro bảo mật mới phát sinh. |
| `codebase-audit-pre-push` | Audit nhanh trước khi push: phát hiện code smell, lỗ hổng, và lỗi tiềm ẩn toàn cục. |
| `fix-review` | Xác nhận patch fix đã xử lý đúng gốc vấn đề và không tạo bug mới. |
| `receiving-code-review` | Dùng khi tiếp nhận feedback review để phản hồi và chỉnh sửa có cơ sở kỹ thuật. |
| `requesting-code-review` | Dùng khi chuẩn bị gửi review để tự kiểm chất lượng trước khi mời reviewer. |
| `security-audit` | Audit bảo mật tổng thể cho module/API/web, phù hợp cho các thay đổi nhạy cảm. |
| `security-auditor` | Chuyên gia rà soát bảo mật theo chiều sâu, ưu tiên lỗ hổng có khả năng khai thác thực tế. |

---

## 2. Tìm diệt bug (Truy xuất rủi ro logic ẩn)

| Skill | Mô tả |
|---|---|
| `find-bugs` | Quét sự thay đổi trong code ở branch hiện tại, dò tìm các lỗi logic lập trình, code quality, hay rủi ro bảo mật nhỏ. |
| `bug-hunter` | Dành khi bạn nghi ngờ code có lỗi logic bên trong nhưng chưa rõ nằm ở đâu. Cung cấp kỹ năng gỡ lỗi (debugging) hệ thống và phân tích từ nguyên nhân gốc rễ (root cause). |
| `error-detective` | Phân tích sự tương quan của code và các dấu vết lỗi trên diện rộng. |
| `debugger` | Kỹ năng debug tổng quát khi đã có triệu chứng lỗi nhưng chưa xác định đúng điểm vỡ. |
| `debugging-strategies` | Chiến lược gỡ lỗi có hệ thống: tái hiện lỗi, khoanh vùng, xác thực giả thuyết, chốt root cause. |
| `systematic-debugging` | Quy trình debug bài bản theo từng bước, phù hợp cho bug khó và lỗi liên quan nhiều module. |
| `lint-and-validate` | Chạy lint/validate sau thay đổi để phát hiện lỗi tĩnh và sai chuẩn trước khi test runtime. |
| `verification-before-completion` | Checklist xác minh trước khi kết luận "đã xong" (test pass, regression check, evidence). |
| `test-driven-development` | Dùng khi muốn bổ sung test theo hướng TDD trước khi sửa hoặc mở rộng logic của module. Phù hợp khi cần khóa hành vi mong đợi bằng test trước. |
| `test-fixing` | Chuyên xử lý test fail theo nhóm nguyên nhân để khôi phục độ ổn định test suite nhanh. |
| `testing-qa` | Góc nhìn QA tổng thể cho unit/integration/e2e nhằm giảm bug lọt production. |
| `unit-testing-test-generate` | Sinh thêm unit test cho hàm, class, hoặc module hiện có. Hữu ích khi muốn mở rộng coverage nhanh nhưng vẫn giữ test dễ bảo trì. |
| `python-testing-patterns` | Mẫu kiểm thử Python thực dụng (pytest, fixtures, mocking) cho codebase hiện tại. |
| `e2e-testing` | Dùng khi cần bổ sung test end-to-end cho luồng web/UI hoàn chỉnh, không chỉ test logic Python thuần ở cấp module. |

---

## 3. Phân tích logic chuyên sâu nâng cao (Academic / Math)

| Skill | Mô tả |
|---|---|
| `matematico-tao` | Khi logic thuật toán liên quan đến đồ thị, xác suất toán học phức tạp cần chứng minh tính đúng đắn chặt chẽ ở cấp độ lý thuyết (đặc biệt trong các thuật toán trading như Pairs Trading). |
| `research-engineer` | Kỹ sư nghiên cứu lý thuyết nghiêm ngặt, chỉ tập trung vào sự đúng đắn về mặt lý thuyết và logic (logic correctness & formal verification), hoàn toàn không quan tâm đến "flair". Tuyệt vời nếu muốn bẻ tận cùng cấu trúc logic. |
| `performance-profiling` | Profiling trước khi tối ưu: xác định đúng bottleneck CPU/RAM/IO thay vì tối ưu cảm tính. |
| `performance-optimizer` | Tối ưu hiệu năng có đo lường trước/sau để chứng minh cải thiện thực tế. |
| `python-performance-optimization` | Tối ưu code Python cho các pipeline xử lý dữ liệu và mô hình trong project. |
| `backtesting-frameworks` | Rà soát logic backtest để giảm bias (look-ahead, survivorship) và sai lệch kết quả. |