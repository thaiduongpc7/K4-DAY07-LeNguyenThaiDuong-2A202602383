# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Nguyễn Thái Dương  
**Nhóm:** Violet  
**Ngày:** 2026-09-19

> Nộp 1 bản / sinh viên. Phần nhóm được tổng hợp trong báo cáo nhóm, còn phần cá nhân tập trung vào quá trình thực hiện, cách tiếp cận, và đánh giá cá nhân của từng người.

**Tổng điểm phần cá nhân: 58** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (8).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### 1.1. Độ tương tự Cosine

**Độ tương tự cosine cao nghĩa là gì?**
Độ tương tự cosine cao cho thấy hai vector embedding có hướng gần nhau, do đó hai đoạn văn có chủ đề hoặc ý nghĩa tương đồng. Giá trị càng gần 1 thì mức tương đồng càng lớn.

**Ví dụ có độ tương tự cao:**
- Câu A: Sinh viên cần nộp hồ sơ học bổng trước thời hạn.
- Câu B: Người học phải gửi đơn xin hỗ trợ tài chính đúng hạn.
- Giải thích: Hai câu diễn đạt cùng một ý tưởng dù dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự thấp:**
- Câu A: Thư viện mở cửa đến 21 giờ.
- Câu B: Cá voi xanh sống ở đại dương.
- Giải thích: Hai câu hoàn toàn không cùng chủ đề.

**Tại sao cosine phù hợp hơn Euclidean distance cho text embeddings?**
Vì cosine quan tâm đến hướng của vector, tức là quan tâm tới ngữ nghĩa hơn là độ dài tuyệt đối của embedding. Trong văn bản, hai câu có thể dài ngắn khác nhau nhưng vẫn mang cùng ý nghĩa; cosine phản ánh điều này tốt hơn khoảng cách Euclidean.

### 1.2. Bài toán tính toán Chunking

**Tài liệu dài 10.000 ký tự, với chunk_size = 500 và overlap = 50. Số chunk là bao nhiêu?**
Ta có công thức:

$$
\left\lceil \frac{10000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = 23
$$

Vậy số chunk là 23. Kết quả này cũng khớp với cách hoạt động của `FixedSizeChunker` trong repo.

**Nếu overlap tăng lên 100, số chunk thay đổi như thế nào?**
Khi overlap = 100, bước nhảy giảm còn 400, nên ta có:

$$
\left\lceil \frac{10000 - 100}{500 - 100} \right\rceil = \left\lceil \frac{9900}{400} \right\rceil = 25
$$

Vì vậy số chunk tăng lên 25. Overlap lớn giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk, nhưng đồng thời làm tăng số lượng chunk, chi phí embedding và dung lượng lưu trữ.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### 2.1. Các hàm chia nhỏ (Chunking functions)

**`SentenceChunker.chunk`**  
Phương pháp của tôi là tách văn bản theo dấu kết thúc câu bằng regex `(?<=[.!?])\s+`. Cách này giúp không làm mất dấu câu, đồng thời phân tách văn bản rõ ràng khi có nhiều câu. Sau đó, các câu được strip và gom thành từng chunk theo `max_sentences_per_chunk`. Mặc dù hiệu quả với văn bản tự nhiên, phương pháp này vẫn có hạn chế với chữ viết tắt và dữ liệu dạng bảng hoặc danh sách.

**`RecursiveChunker.chunk` / `_split`**  
Cách tiếp cận ở đây là ưu tiên chia theo các separator có mức độ “ngữ nghĩa” cao nhất trước, ví dụ khoảng trắng giữa đoạn, newline, dấu chấm câu, rồi đến khoảng trắng đơn. Nếu một mảnh vẫn quá dài, hàm sẽ lùi xuống mức separator nhỏ hơn và tiến hành đệ quy. Đây là cách rất hợp lý cho dữ liệu có nhiều cấu trúc đoạn văn, danh sách và heading.

### 2.2. Lớp `EmbeddingStore`

**`add_documents` + `search`**  
Mỗi `Document` được sao chép metadata, sinh embedding và lưu vào store bộ nhớ. Khi tìm kiếm, query cũng được chuyển thành embedding, sau đó so sánh với các vector đã lưu bằng dot product. Trong benchmark này, vector mock đã được chuẩn hóa nên dot product gần tương đương cosine similarity. Đây là cách đơn giản nhưng rất phù hợp cho môi trường lab.

**`search_with_filter` + `delete_document`**  
`search_with_filter` lọc theo metadata trước, rồi mới tính độ tương đồng trên tập ứng viên còn lại. `delete_document` xóa tất cả record có cùng `doc_id` và trả về `True` nếu có dữ liệu bị xóa. Cách này giúp thao tác với dữ liệu ngắn gọn, dễ kiểm tra và phù hợp với nhu cầu của bài lab.

### 2.3. Tác tử `KnowledgeBaseAgent`

**`answer`**  
Agent lấy top-k chunk phù hợp nhất, đánh số từng chunk và đưa cả nội dung lẫn nguồn vào prompt. Sau đó, prompt yêu cầu chỉ sử dụng ngữ cảnh có sẵn, nói rõ khi không tìm thấy thông tin, đồng thời trích dẫn số chunk để dễ kiểm tra grounding. Với store rỗng, agent xử lý được mà không cần gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Đây là phần tôi dành nhiều thời gian nhất, vì phần này quyết định khả năng chạy đúng các bài test và khả năng hoạt động của hệ thống RAG.

### Kết quả kiểm thử

```text
============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua:** 42 / 42.

Từ đó, tôi xác nhận các phần chính như chunking, store, agent và các chức năng phụ đã hoạt động ổn định theo yêu cầu của đề bài.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Quy tắc phân loại được đặt trước khi đọc kết quả: score >= 0.5 là cao, score < 0.5 là thấp. Vì benchmark dùng `_mock_embed`, đây chỉ là quy ước để quan sát pipeline, không phải cách đo ngữ nghĩa thực sự.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-------|-------|---------|--------------|-------|
| 1 | Sinh viên cần nộp hồ sơ học bổng trước thời hạn. | Người học phải gửi đơn xin hỗ trợ tài chính đúng hạn. | Cao | -0.051912 | Không |
| 2 | Thư viện mở cửa đến 21 giờ. | Cá voi xanh sống ở đại dương. | Thấp | -0.011932 | Có |
| 3 | Học bổng có giá trị 5 triệu đồng. | Mức hỗ trợ là 5.000.000 VNĐ. | Cao | -0.080586 | Không |
| 4 | Sinh viên phải đạt điểm rèn luyện loại tốt. | Sinh viên phải đạt điểm rèn luyện loại yếu. | Thấp | 0.190787 | Có |
| 5 | Hạn nộp hồ sơ là 30/06/2026. | Thư viện mở cửa đến 21 giờ. | Thấp | -0.024239 | Có |

**Nhận xét:**
Cặp 1 và cặp 3 có cùng ý nghĩa gần như tương đương nhưng lại cho điểm âm, trong khi cặp 4 tuy có từ vựng tương đồng nhưng mang ý nghĩa trái ngược lại lại có điểm cao hơn. Điều này cho thấy `_mock_embed` không mô hình hóa đúng ngữ nghĩa, mà chủ yếu phục vụ kiểm tra cấu trúc dữ liệu và pipeline. Nói cách khác, benchmark này giúp kiểm tra khả năng chạy code, không phải đánh giá chất lượng semantic embedding thực sự.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (8 điểm)

Sau khi tối ưu ranking theo intent và evidence phrase, tôi đã chạy lại benchmark thực tế trên pipeline hiện tại của `src` và kiểm tra rõ ràng các gold doc / evidence string trong top-3.

| # | Câu hỏi | Gold doc | Evidence trong top-3? | Kết quả thực tế |
|---|----------|----------|------------------------|-----------------|
| 1 | Học bổng khuyến khích học tập ở VIMARU được xét như thế nào? | `vimaru-hbkkht-tieu-chuan-sinh-vien` | Có | `naive=2`, `content=2` |
| 2 | Học bổng Vallet sau đại học năm 2026 có bao nhiêu suất và trị giá bao nhiêu? | `vallet-hoc-bong-sau-dai-hoc` | Có | `naive=2`, `content=2` |
| 3 | Quy trình xét học bổng hỗ trợ đột xuất của UEH gồm những bước nào và mất bao lâu? | `ueh-ke-hoach-xet-hoc-bong-2026` | Có | `naive=2`, `content=2` |
| 4 | Hồ sơ đăng ký học bổng K-T của ULIS gồm những giấy tờ gì? | `ulis-hoc-bong-kt-2025-2026` | Có | `naive=2`, `content=2` |
| 5 | Tân sinh viên HSB muốn được tài trợ 100% học phí có điều kiện thì cần điểm thi bao nhiêu và phải hoàn trả thế nào? | `hsb-hoc-bong-tan-sinh-vien-2026` | Chưa | `naive=2`, `content=0` |

**Nhận xét chung:**
- Kết quả thực tế sau tối ưu là `naive = 10/10` và `content = 8/10`.
- Điều này cho thấy hệ thống hiện đã tìm đúng gold document ở hầu hết các câu hỏi, và phần lớn các câu hỏi đã lấy được đoạn evidence cần thiết trong top-3.
- Q5 vẫn là điểm nghẽn cuối cùng: mặc dù tài liệu HSB nằm ở top-1 theo gold doc, cụm evidence `24/30` chưa xuất hiện trong top-3 retrieved context, nên phần content điểm chưa đạt 2/2.
- Đây cũng là dấu hiệu thực tế cho thấy, với mock embedding, ranking cuối cùng vẫn cần sự hỗ trợ từ exact phrase matching và metadata cho những câu hỏi yêu cầu số liệu rất cụ thể.

**Backend và chiến lược chạy benchmark:**
- `mock embeddings`
- `RecursiveChunker(chunk_size=500)`
- Tổng cộng 109 chunks
- Kết quả thực tế từ `bench.py`: `naive = 10/10`, `content = 8/10`.

**Điều học được từ nhóm:**
Việc cải thiện retrieval không chỉ dựa trên embedding mà còn phụ thuộc rất lớn vào việc nhận diện đúng evidence phrase liên quan đến số liệu, giấy tờ, thời gian và điều kiện. Đây là bước quan trọng để biến một hệ thống retrieval “đúng tài liệu” thành một hệ thống “đúng câu trả lời”.

---

## Tự đánh giá cá nhân

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **58 / 60** |

