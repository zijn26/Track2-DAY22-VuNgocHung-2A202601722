# Báo Cáo Bằng Chứng (Evidence Report) — Day 22 Lab

## 📁 Danh Sách Tệp Bằng Chứng (Evidence Files)

| Tệp Bằng Chứng | Mô Tả | Trạng Thái |
|---|---|---|
| `01_langsmith_traces.png` | Ảnh chụp màn hình giao diện LangSmith Dashboard hiển thị ≥ 50 traces | 📌 Ảnh màn hình từ web |
| `02_prompt_hub.png` | Ảnh chụp màn hình giao diện LangSmith Prompt Hub hiển thị 2 phiên bản prompt | 📌 Ảnh màn hình từ web |
| `02_ab_routing_log.txt` | Console log kết quả chạy A/B Routing cho 50 câu hỏi | ✅ Đã lưu |
| `03_ragas_scores.png` | Ảnh chụp terminal hiển thị bảng điểm so sánh RAGAS của V1 và V2 | 📌 Ảnh màn hình terminal |
| `03_ragas_report.json` | Tệp báo cáo JSON chi tiết kết quả đánh giá RAGAS | ✅ Đã lưu (`data/ragas_report.json` & `evidence/03_ragas_report.json`) |
| `04_pii_demo_log.txt` | Console log kết quả demo kiểm duyệt và redact PII với Guardrails | ✅ Đã lưu |
| `04_json_demo_log.txt` | Console log kết quả demo sửa và định dạng JSON với Guardrails | ✅ Đã lưu |

---

## 📊 Kết Quả Đánh Giá RAGAS (RAGAS Evaluation Results)

| Metric | Prompt V1 (Ngắn gọn) | Prompt V2 (Có cấu trúc) | Winner | Trạng Thái Mục Tiêu |
|---|---|---|---|---|
| **Faithfulness (Tính trung thực)** | **0.9335** | **0.9466** | **← V2** | ⭐ Đạt mục tiêu (≥ 0.8 & ≥ 0.9) |
| **Answer Relevancy (Độ liên quan)** | **0.9158** | 0.8968 | **← V1** | ⭐ Đạt mức rất cao (> 0.89) |
| **Context Recall (Độ thu hồi context)** | **1.0000** | **1.0000** | **Hòa (1.0)** | ⭐ Hoàn hảo (100%) |
| **Context Precision (Độ chính xác context)**| **0.9483** | 0.9417 | **← V1** | ⭐ Đạt mức rất cao (> 0.94) |

---

## 🔍 Phân Tích Chuyên Sâu Kết Quả V1 so với V2 (Analytical Report)

1. **Faithfulness (Tính trung thực — V2 thắng với 0.9466 vs 0.9335):**
   - **Lý do V2 cao hơn**: Prompt V2 được thiết kế với phong cách chuyên gia (*Expert tone*), yêu cầu mô hình đọc kỹ thông tin, liệt kê sự thật (*facts*) và có cấu trúc logic rõ ràng từ 3-5 câu. Điều này giúp LLM hạn chế tối đa suy diễn ảo (hallucination) và bám sát chính xác từng chi tiết có trong tài liệu `context`.
   - **Điểm thưởng đặc biệt**: Cả V1 và V2 đều vượt xa mốc chuẩn 0.80 và đạt trên **0.90** (`V1=0.9335`, `V2=0.9466`).

2. **Answer Relevancy (Độ liên quan — V1 thắng với 0.9158 vs 0.8968):**
   - **Lý do V1 cao hơn**: Prompt V1 yêu cầu trả lời ngắn gọn (2-4 câu), đi thẳng vào trọng tâm câu hỏi của người dùng. Việc ngắn gọn giúp giảm các từ ngữ thừa, tăng mật độ từ khóa trực tiếp liên quan đến câu hỏi.

3. **Context Recall & Precision (1.0000 & ~0.945):**
   - **Context Recall đạt 1.0000 (100%)**: Hệ thống FAISS Retriever với `k=3` cùng bộ phân đoạn `chunk_size=500` thu hồi được 100% lượng kiến thức cần thiết để trả lời đúng cho tất cả 50 cặp câu hỏi test.
   - **Context Precision đạt >0.94**: Các đoạn tài liệu liên quan nhất luôn được xếp ở vị trí đầu tiên trong danh sách truy xuất.
