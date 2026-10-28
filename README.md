# DS201 - Lab 2: Mạng Neural Tích Chập (Convolutional Neural Networks)

Báo cáo và mã nguồn thực hành Lab 2 môn **DS201 - Học sâu (Deep Learning)**.

Bài thực hành tập trung vào việc tự xây dựng (from scratch), huấn luyện và đánh giá các kiến trúc CNN kinh điển trên bộ dữ liệu **VinaFood21** (21 lớp món ăn Việt Nam).

---

## Nội dung bài thực hành

Notebook [`DS201_Lab2_Assignment.ipynb`](DS201_Lab2_Assignment.ipynb) gồm 3 phần bài tập chính:

### 1. Bài 1 - LeNet-5: Kiến trúc CNN kinh điển & Pooling
- **Kiến trúc**: Xây dựng mô hình LeNet-5 tùy biến cho ảnh màu kích thước `64 × 64 × 3` (tổng tham số: 339,421; số chiều sau flatten: 2,704).
- **Thực nghiệm & So sánh**:
  - Huấn luyện mô hình LeNet-5 sử dụng **Average Pooling (`nn.AvgPool2d`)** (nguyên bản của LeNet).
  - Huấn luyện mô hình LeNet-5 sử dụng **Max Pooling (`nn.MaxPool2d`)**.
- **Đánh giá**: So sánh định lượng hiệu quả giữa hai cơ chế pooling trên cùng tập dữ liệu; trực quan hóa đồ thị hàm mất mát (loss curve), ma trận nhầm lẫn (Confusion Matrix) và báo cáo phân loại chi tiết (Classification Report).

### 2. Bài 2 - GoogLeNet rút gọn (GoogLeNetSmall): Multi-scale Feature Extraction
- **Khối InceptionBlock**: Cài đặt khối Inception 4 nhánh song song:
  - Nhánh 1: $1 \times 1$ Conv
  - Nhánh 2: $1 \times 1$ Conv (giảm chiều) $\rightarrow$ $3 \times 3$ Conv
  - Nhánh 3: $1 \times 1$ Conv (giảm chiều) $\rightarrow$ $5 \times 5$ Conv
  - Nhánh 4: $3 \times 3$ MaxPool $\rightarrow$ $1 \times 1$ Conv
- **Kiến trúc GoogLeNetSmall**: Thiết kế cho ảnh `64 × 64 × 3`, bao gồm:
  - Tầng Stem trích xuất đặc trưng ban đầu (`Conv 3x3` $\rightarrow$ `MaxPool 2x2`).
  - 3 khối Inception lồng ghép với số kênh tăng dần: 56 $\rightarrow$ 112 $\rightarrow$ 144 kênh xen kẽ các tầng Max Pooling.
  - Tầng phân loại sử dụng **Adaptive Average Pooling** (Global Pooling) kết hợp một tầng Fully Connected.
- **Đánh giá**: Đo đạc độ chính xác và macro F1-score trên tập kiểm thử (Test set), trực quan hóa đồ thị huấn luyện.

### 3. Bài 3* - ResNet-18: Kết nối phần dư & Data Augmentation
- **Khối BasicBlock (From Scratch)**: Cài đặt khối phần dư gồm 2 tầng Conv $3 \times 3$, Batch Normalization, ReLU và đường tắt (skip connection / shortcut). Hỗ trợ projection shortcut khi thay đổi kích thước hoặc số kênh (`stride != 1` hoặc `in_ch != out_ch`).
- **Kiến trúc ResNet-18 hoàn chỉnh**:
  - Tầng Stem: Conv $7 \times 7$ (stride 2) + BatchNorm + ReLU + MaxPool $3 \times 3$ (stride 2).
  - 4 stages tương ứng cấu hình `[2, 2, 2, 2]` BasicBlocks với các kênh `64, 128, 256, 512`.
  - Tầng Adaptive Average Pooling và Fully Connected phân loại vào 21 lớp.
  - Tổng số tham số kiểm định: đúng **11,187,285 tham số**.
  - Kích thước ảnh đầu vào: `96 × 96 × 3`.
- **Thực nghiệm & So sánh**:
  - **Lần 1 - Không áp dụng Data Augmentation**: Chỉ Resize và Normalize.
  - **Lần 2 - Có áp dụng Data Augmentation**: `RandomResizedCrop(scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter()`.
- **Mục tiêu phân tích**: Đánh giá định lượng tác động của tăng cường dữ liệu trong việc giảm thiểu hiện tượng quá khớp (overfitting) trên mạng nơ-ron sâu.

---

## Bộ dữ liệu VinaFood21

- **Mô tả**: Bộ dữ liệu gồm 21 lớp món ăn Việt Nam được chia thành 2 tập chính: `train/` và `test/`.
- **Phân chia tập dữ liệu**:
  - Nếu đã có sẵn thư mục `val/`, notebook sẽ tự động nạp trực tiếp.
  - Nếu chưa có, notebook tự động trích xuất **15% dữ liệu từ tập `train/`** làm tập validation với seed cố định (`SEED = 42`).
- **Tự động tải dữ liệu**: Cell 4 của notebook hỗ trợ tự động tải file zip qua Google Drive ID (`1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR`) và giải nén tự động.
- **Chuẩn hóa ảnh**: Chuẩn hóa theo thống kê phân phối ImageNet:
  - `MEAN = [0.485, 0.456, 0.406]`
  - `STD  = [0.229, 0.224, 0.225]`
- **Cấu trúc thư mục dữ liệu**:
```text
VinaFood21/
├── train/
│   ├── Banh beo/
│   ├── Banh mi/
│   └── ... (21 thư mục lớp)
└── test/
    ├── Banh beo/
    ├── Banh mi/
    └── ... (21 thư mục lớp)
```

---

## Cấu hình huấn luyện chung (Hyperparameters)

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Áp dụng cho DataLoader train, val và test |
| **Epochs** | 20 | Số vòng lặp huấn luyện cho mỗi thực nghiệm |
| **Optimizer** | Adam | Tốc độ học ban đầu `lr = 1e-3` |
| **Loss Function** | CrossEntropyLoss | Hàm mất mát phân loại đa lớp |
| **Random Seed** | 42 | Đảm bảo tính tái lập (reproducibility) |
| **Image Size** | 64 × 64 (Bài 1, 2) / 96 × 96 (Bài 3) | Kích thước ảnh đưa vào mô hình |
| **Chỉ số đánh giá** | Accuracy, Macro Precision, Macro Recall, Macro F1 | Tính toán qua thư viện `sklearn.metrics` |

---

## Cấu trúc thư mục

```text
ds201_lab2/
├── .gitignore                 
├── DS201_Lab2_Assignment.ipynb    
├── requirements.txt              
├── 24521991.txt                     
└── README.md                       
```

---

## Hướng dẫn thực thi

### Cách 1: Chạy trên Google Colab (Khuyến nghị)

1. Mở file `DS201_Lab2_Assignment.ipynb` bằng Google Colab.
2. Chuyển đổi môi trường thực thi sang **GPU**:
   - Menu: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** (hoặc GPU tương đương).
3. Chạy tuần tự các ô lệnh từ trên xuống:
   - Cell mục *1. Import dữ liệu* đã tích hợp sẵn đoạn mã tải tự động `VinaFood21.zip` qua `gdown` từ Google Drive và giải nén trực tiếp vào `./VinaFood21`.
   - Kết quả đánh giá, đồ thị loss và bảng chỉ số metric sẽ được sinh ra trực tiếp theo thời gian thực.

### Cách 2: Chạy trên máy cá nhân (Local Environment)

1. **Khởi tạo môi trường ảo Python**:
   ```bash
   python -m venv .venv
   # Trên Windows:
   .venv\Scripts\activate
   # Trên Linux/macOS:
   source .venv/bin/activate
   ```

2. **Cài đặt các gói phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu**:
   - Đặt thư mục `VinaFood21/` hoặc giải nén `VinaFood21.zip` ngay trong thư mục `ds201_lab2/`.

4. **Khởi chạy Jupyter Notebook**:
   ```bash
   jupyter notebook DS201_Lab2_Assignment.ipynb
   ```

