# Nested Learning Demo

Demo minh họa **Nested Learning** - một paradigm học tập mới từ Google Research cho Học Lồng Ghép.

## Paper gốc

- **Tên:** "Nested Learning: The Illusion of Deep Learning Architectures"
- **Hội nghị:** NeurIPS 2025
- **Tác giả:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)
- **Blog:** <https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/>

## Nội dung

| File | Mô tả |
|------|-------|
| `nested_learning_demo.ipynb` | Notebook Colab với lý thuyết + code demo |
| `nested_learning_demo.py` | Script Python độc lập |
| `Nested Learning - Học Lồng Ghép.pdf` | Tài liệu Việt hóa |

## 4 điểm mạnh chính

1. **Catastrophic Forgetting** - Giải quyết vấn đề quên kiến thức cũ khi học task mới
2. **Deep Optimizers** - Optimizer là memory module nén gradients
3. **Continuum Memory System** - Bộ nhớ đa tần suất (fast/medium/slow)
4. **Unified Architecture & Optimization** - Thống nhất kiến trúc và tối ưu hóa

## Cập nhật kỹ thuật (Technical Updates)

- **Kiến trúc mạng:** Sử dụng **ReLU** activation và **64 hidden units** (thay vì Tanh/16 units) để có thể học được các đường biên phức tạp (như hình tròn).
- **Cơ chế bảo tồn (Consolidation):** Cài đặt cơ chế **Elastic Weight Consolidation (EWC)** đơn giản hóa.
  - Khi chuyển task, mô hình lưu lại "snapshot" trọng số cũ.
  - Thêm một `reg_strength` (lực kéo) vào hàm loss để ngăn trọng số mới đi quá xa so với trọng số cũ.
  - Hiện tại `reg_strength = 15.0` để cân bằng giữa việc nhớ cũ (Stability) và học mới (Plasticity).

## Dữ liệu đầu vào (Input Data)

Demo sử dụng 3 bài toán phân loại (Classification Tasks) khác nhau được học tuần tự để kiểm tra khả năng ghi nhớ:

1. **Task 1 - Linear Pattern:** Phân loại tuyến tính đơn giản.
   - Quy luật: `y = 1` nếu `2*x1 + 0.5*x2 > 0`
2. **Task 2 - XOR Pattern:** Phân loại phi tuyến tính (bàn cờ).
   - Quy luật: `y = 1` nếu `x1 * x2 > 0` (Góc phần tư 1 và 3)
3. **Task 3 - Circular Pattern:** Phân loại hình tròn.
   - Quy luật: `y = 1` nếu `x1^2 + x2^2 < 1` (Bên trong hình tròn bán kính 1)

```python
# Code minh họa dữ liệu (2 chiều)
# Input X: [x1, x2] ngẫu nhiên từ phân phối chuẩn
X = np.random.randn(100, 2)

# Task 1: Linear
y_linear = (2 * X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)

# Task 2: XOR
y_xor = ((X[:, 0] * X[:, 1]) > 0).astype(float)

# Task 3: Circular
y_circle = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(float)
```

Mô hình sẽ học lần lượt Task 1 -> Task 2 -> Task 3. Thử thách là sau khi học Task 3, mô hình có còn nhớ cách giải Task 1 không?

## Kết quả demo

### 1. Decision Boundaries (trực quan hóa vùng kiến thức)

So sánh khả năng ghi nhớ kiến thức cũ giữa mạng thường và Nested Learning qua các task liên tiếp.

![Decision Boundaries](nested_learning_boundaries.png)

> **Nhận xét:**
> - **Hàng trên (Simple Network):** Khi chuyển sang học Task 2 (XOR), mô hình "quên sạch" Task 1 (Linear). Vùng phân chia (màu xanh/đỏ) thay đổi hoàn toàn để phục vụ task mới nhất. Đây là hiện tượng *Catastrophic Forgetting*.
> - **Hàng dưới (Nested Learning):** Khi học Task 2 và 3, mô hình vẫn cố gắng duy trì cấu trúc phân chia của các task cũ.
>   - Ở Task 3 (Circular), dù phải học hình tròn mới, bạn vẫn thấy bóng dáng các vệt màu của Task 2 (XOR) và Task 1 (Linear) ở nền.
>   - Điều này chứng tỏ kiến thức cũ được bảo tồn nhờ cơ chế Regularization (dây thun ký ức).

### 2. Performance & Memory (hiệu năng và bộ nhớ)

Chi tiết về độ chính xác, hoạt động của Deep Optimizer và Continuum Memory System.

![Performance Demo](nested_learning_demo.png)

> **Nhận xét:**
> - **Biểu đồ 1 (Accuracy):** Cột màu xanh (Nested) luôn cao hơn màu đỏ (Simple) ở cuối quá trình huấn luyện, cho thấy khả năng học task mới mà không đánh đổi kiến thức cũ.
> - **Biểu đồ 4 (Memory System):** Đường màu xanh lá (Slow Memory) thay đổi rất chậm và mượt mà, đóng vai trò như bộ nhớ dài hạn. Đường màu đỏ (Fast Memory) bám sát tín hiệu đầu vào, đóng vai trò xử lý thông tin tức thời.

### 3. Bảng đánh giá hiệu suất (Quantitative Evaluation)

Kết quả chạy script `nested_learning_evaluation.py` trên tập dữ liệu kiểm thử (200 samples/task):

| Model | Avg Accuracy | Backward Transfer (BWT) |
|-------|--------------|-------------------------|
| **Simple Network** | 63.67% | -33.50% (Quên nhiều) |
| **Nested Learning** | **65.83%** | **-27.75%** (Quên ít hơn) |

**Chi tiết độ chính xác sau khi học xong Task 3:**

| Task | Simple Network | Nested Learning | Nhận xét |
|------|----------------|-----------------|-----------|
| Task 1 (Linear) | 51.50% | 48.50% | Cả 2 đều gặp khó khăn với task đầu tiên |
| Task 2 (XOR) | 66.00% | **74.50%** | **Nested bảo tồn kiến thức tốt hơn hẳn (+8.5%)** |
| Task 3 (Circular)| 73.50% | **74.50%** | Nested học task mới vẫn tốt |

> **Kết luận:** Nested Learning chấp nhận hy sinh một chút ở task quá xa (Task 1) để giữ lại rất tốt kiến thức của task gần nhất (Task 2), giúp tổng thể hệ thống (Avg Accuracy) ổn định hơn và ít bị "quên" (BWT cao hơn).

## Chạy demo

### Google Colab

Mở `nested_learning_demo.ipynb` trên Google Colab và chạy từng cell.

### Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
python nested_learning_demo.py
```

## Việt hóa

- Bùi Huỳnh Kinh Luân - <luanbhk@gmail.com>
