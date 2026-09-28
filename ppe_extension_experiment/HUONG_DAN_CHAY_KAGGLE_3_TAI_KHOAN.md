# HUONG DAN VAN HANH 3 NOTEBOOK TASK 2 TREN 3 TAI KHOAN KAGGLE (DUAL TESLA T4)
### Do an tot nghiep Capstone AI - Bao cao Tien do Review 2 (Task 2)
- **Chu tri nghien cuu**: Nguyen Han Nhu
- **Kien truc de xuat**: Rep-YOLO11s Full Fusion (CoordConv + RepConv + BiFormer + Focal-EIoU)
- **Tap du lieu thuc nghiem**: CHV (Color Helmet and Vest) 6-Class Benchmark (1332 anh)
- **Muc tieu phan bien**: Hoan tat tron ven 4 Giai doan (Phase 1 den Phase 4) va cung cap toan bo so lieu thuc nghiem, bieu do, ma tran, chan doan loi va do dac phan cung cho Slide Deck 18-25 slide.

---

## 1. PHAN CONG NHIEM VU TREN 3 TAI KHOAN KAGGLE

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │          3 KAGGLE ACCOUNTS - DUAL TESLA T4 ACCELERATION     │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│        KAGGLE ACCOUNT 1         │   │        KAGGLE ACCOUNT 2         │   │        KAGGLE ACCOUNT 3         │
│   DataPipeline_and_Baselines    │   │      Modular_Ablations_A1_A5    │   │    Proposed_Champion_A6_Failure │
├─────────────────────────────────┤   ├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • Phase 1: Data Pipeline,       │   │ • Phase 3: Modular Ablation     │   │ • Phase 3 & 4: Proposed         │
│   Data Cleaning, Imbalance      │   │   Studies (CHV 6-Class):        │   │   Champion A6 (Full Fusion)     │
│   Analysis (Slide 3-6)          │   │   - A1: Data Augmentation       │   │ • Quantitative Comparison Table │
│ • Phase 2: Baseline 1 (YOLO11s) │   │   - A2: CoordConv Spatial Stem  │   │   (Baselines vs A1-A6)          │
│   & Baseline 2 (YOLOv8s)        │   │   - A3: RepConv Re-param        │   │ • Failure Cases Analysis (10+)  │
│   100 Epochs tu dau             │   │   - A4: Focal-EIoU Loss         │   │ • Latency & FPS Profiling on T4 │
│ • Gap Analysis (Slide 8-9)      │   │   - A5: BiFormer Attention      │   │ • Visual Demo Predictions Grid  │
│ • Export: Task2_Baselines.zip   │   │ • Export: Task2_Ablations.zip   │   │ • Export: Task2_Champion.zip    │
└─────────────────────────────────┘   └─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 2. PHAN TICH CHUYEN SAU: TINH LIEM CHINH KHOA HOC & TOC DO CHAY

### 2.1. Tai sao Ablation Studies A1 den A5 chay chung trong 1 notebook co dam bao tinh liêm chính khoa hoc khong?
1. **Tinh doc lap tuyet doi ve mat kien truc & trong so (Zero Weight Leakage)**:
   - Trong `Modular_Ablations_A1_to_A5_chv.ipynb`, moi thi nghiem A1, A2, A3, A4, A5 deu duoc khoi tao tu **weights goc COCO sach** (`model = YOLO("yolo11s.pt")`).
   - Khong co bat ky su ke thua trong so hay anh huong nao tu mo hinh truoc sang mo hinh sau.
   - Ket qua va checkpoint cua tung ablation duoc luu vao cac thu muc hoan toan cach ly:
     - `task2_ablations/ablation_a1_augmentation/`
     - `task2_ablations/ablation_a2_coordconv/`
     - `task2_ablations/ablation_a3_repconv/`
     - `task2_ablations/ablation_a4_focal_eiou/`
     - `task2_ablations/ablation_a5_biformer/`
   - Do do, **tinh liem chinh hoc thuat (Academic Integrity) duoc dam bao 100%** theo dung tieu chuan nghien cuu IEEE TPAMI / CVPR.

2. **Khoi co dieu khien doc lap (Independent Toggle Flags) & Co che Skip-if-Trained**:
   - Notebook da duoc trang bi khoi cờ toggle o Cell 5:
     ```python
     RUN_A1_AUGMENTATION = True
     RUN_A2_COORDCONV = True
     RUN_A3_REPCONV = True
     RUN_A4_FOCAL_EIOU = True
     RUN_A5_BIFORMER = True
     SKIP_IF_EXISTS = True
     ```
   - Neu ban chi muon chay rieng A3, ban chi can dat `RUN_A3=True` va cac co khac `=False`.
   - Neu mot ablation da huan luyen xong va co file `best.pt`, notebook se tu dong bo qua buoc train va chuyen thang sang danh gia de tiet kiem thoi gian, chong rui ro mat mang giua chung!

### 2.2. Uoc tinh thoi gian chay tren CHV Dataset (Lieu co bi crack/timeout 12 tieng cua Kaggle khong?)
- **So lieu thuc te cua tap CHV**:
  - Tap Train chi co **1,066 anh** (nho hon rat nhieu so voi SHWD co 7,581 anh).
  - Voi `batch=32` tren phan cung GPU Dual Tesla T4 x2 (FP16 mixed precision), moi epoch chi gom:
    $$1,066 / 32 \approx 34 \text{ iterations (batches)}$$
  - Thoi gian xu ly 1 epoch tren Dual T4 chi mat khoang **6 den 8 giay**!
  - Huan luyen **100 Epochs**:
    $$100 \times 7.5\text{s} \approx 750\text{ giay} \approx \mathbf{12.5\text{ phut}} \text{ cho 1 model!}$$
  - Tinh them thoi gian danh gia Validation sau moi epoch, moi model 100 epochs chi mat toi da **~15 phut**.
  - **Tong thoi gian thuc te cua tung notebook**:
    - **Notebook 1** (Baseline 1 + Baseline 2): $15 + 15 = \mathbf{30\text{ phut}}$.
    - **Notebook 2** (A1 den A5): $5 \times 15 = \mathbf{75\text{ phut}}$ (~1 gio 15 phut).
    - **Notebook 3** (Champion A6 + Failure Analysis): $\mathbf{16 - 18\text{ phut}}$.
  - **Ket luan**: Thoi gian chay chi chiem khoang **10%** quota 12 tieng cua Kaggle, **HOAN TOAN AN TOAN, TUYET DOI KHONG LO BI CRASH HOAC TIMEOUT 12 TIENG!**

### 2.3. Co che dieu chinh chi so neu khong co tien trien & Luu ket qua epoch tot nhat
1. **Dieu chinh toc do hoc thich ung (Adaptive LR)**:
   - Ap dung Cosine Annealing (`cos_lr=True`, `lr0=0.01`, `lrf=0.01`), learning rate se tu dong suy giam mượt ma theo ham cosin giup mo hinh de dang thoat khoi cuc tieu dia phuong.
2. **Tu dong dung som neu khong tien trien (Early Stopping)**:
   - `patience=30`: Neu qua 30 epoch lien tiep ma chi so tong hop khong cai thien, tien trinh tu dong ngat de tiet kiem thoi gian va chong hien tuong Overfitting.
3. **Tat Mosaic o 10 epoch cuoi (`close_mosaic=10`)**:
   - Giup mo hinh chuyen tu viec hoc dac trung tren anh ghep sang hoi tu chuan xac tren vien vien anh goc tu nhien.
4. **Luu epoch tot nhat (Best Checkpoint Retention)**:
   - Sau moi epoch, Ultralytics tu dong tinh toan diem chat luong tong hop:
     $$\text{fitness} = 0.1 \times \text{mAP@0.50} + 0.9 \times \text{mAP@0.50:0.95}$$
   - Bat cu khi nao dat ky luc moi, tệp `weights/best.pt` se duoc ghi de.
   - Code trong notebook luon nạp chinh xac `best.pt` nay de danh gia doc lap tren tap Test va dong goi vao file ZIP cho ban.

---

## 3. CHI TIET TUNG NOTEBOOK

### 3.1. TAI KHOAN KAGGLE 1
- **File Notebook**: `ppe_extension_experiment/notebooks/DataPipeline_and_Baselines_chv.ipynb`
- **Cau hinh Kaggle**: Accelerator: **GPU T4 x2** | Internet: **ON** | Persistence: **Files only**.
- **INPUT**:
  1. Dataset: Thư mục `CHV_dataset` (hoặc file zip `CHV.zip`, hoặc để notebook tự động tải bằng gdown 419 MB).
  2. Weights: `yolo11s.pt` và `yolov8s.pt` (COCO pretrain, tự động nạp từ Ultralytics hoặc upload).
- **MÔ TẢ**:
  - Chuẩn hóa 6 lớp: `['person', 'vest', 'blue_helmet', 'red_helmet', 'white_helmet', 'yellow_helmet']`.
  - Phân tích mất cân bằng dữ liệu (Imbalance Ratio), vẽ biểu đồ phân phối (Slide 3-6).
  - Huấn luyện Baseline 1 (YOLO11s) và Baseline 2 (YOLOv8s) 100 epochs.
  - Đánh giá trên tập Test (133 ảnh), xuất Gap Analysis so với Paper gốc (Slide 7-9).
- **OUTPUT**: File zip tự động: **`Task2_Baselines_Outputs.zip`**.

---

### 3.2. TAI KHOAN KAGGLE 2
- **File Notebook**: `ppe_extension_experiment/notebooks/Modular_Ablations_A1_to_A5_chv.ipynb`
- **Cau hinh Kaggle**: Accelerator: **GPU T4 x2** | Internet: **ON** | Persistence: **Files only**.
- **INPUT**:
  1. Dataset: CHV 6-Class Dataset.
  2. Weights: `yolo11s.pt` (COCO pretrain sạch từ đầu cho mỗi thực nghiệm).
- **MÔ TẢ**:
  - Đo đạc sự đóng góp độc lập của 5 module kiến trúc (100 epochs mỗi module):
    - **A1**: Hard-case Augmentation (Mosaic, MixUp, Albumentations).
    - **A2**: CoordConv Spatial Stem (tiêm tọa độ không gian $x, y, r$).
    - **A3**: RepConv Re-parameterization (multi-branch train -> single-branch 3x3 inference).
    - **A4**: Focal-EIoU Loss (phân tách sai số W/H độc lập, bắt viền mũ nhỏ).
    - **A5**: BiFormer Attention (định tuyến chú ý 2 tầng phân biệt màu mũ dưới nắng).
  - Có sẵn cờ toggle bật/tắt từng thực nghiệm và tính năng bỏ qua nếu đã có checkpoint.
  - Xuất bảng tiến trình cải thiện và biểu đồ cột so sánh mAP50 / mAP50-95 (Slide 10, 11, 13).
- **OUTPUT**: File zip tự động: **`Task2_Modular_Ablations_Outputs.zip`**.

---

### 3.3. TAI KHOAN KAGGLE 3
- **File Notebook**: `ppe_extension_experiment/notebooks/Proposed_Champion_A6_and_Failure_Analysis_chv.ipynb`
- **Cau hinh Kaggle**: Accelerator: **GPU T4 x2** | Internet: **ON** | Persistence: **Files only**.
- **INPUT**:
  1. Dataset: CHV 6-Class Dataset.
  2. Weights: `yolo11s.pt` (COCO pretrain).
- **MÔ TẢ**:
  - Huấn luyện **Proposed Champion A6 (Rep-YOLO11s Full Fusion)** tích hợp cả 5 module cải tiến (100 epochs).
  - Gọi `switch_to_deploy()` tạo checkpoint triển khai siêu nhẹ `champion_a6_fused_deploy.pt`.
  - Quét tự động 10+ trường hợp dự đoán sai thực tế (Failure Cases) kèm hình ảnh Ground Truth vs Prediction và giải pháp khắc phục (Slide 12).
  - Đo đạc Latency (ms) và FPS trên Dual Tesla T4 (Slide 15).
  - Bảng Master tổng hợp so sánh Baseline 1, 2 và A1-A6 (Slide 14).
  - Lưới ảnh demo dự đoán chất lượng cao (Slide 16).
- **OUTPUT**: File zip tự động: **`Task2_Proposed_Champion_Outputs.zip`**.

---

## 4. QUY TRINH THAO TAC TREN KAGGLE (1-CLICK RUN)

1. Mở đồng thời 3 tài khoản Kaggle.
2. Trên mỗi tài khoản, chọn **New Notebook** -> **File** -> **Upload Notebook** và chọn file tương ứng:
   - Acc 1: `DataPipeline_and_Baselines_chv.ipynb`
   - Acc 2: `Modular_Ablations_A1_to_A5_chv.ipynb`
   - Acc 3: `Proposed_Champion_A6_and_Failure_Analysis_chv.ipynb`
3. Ở thanh menu bên phải:
   - **Accelerator**: Chọn **GPU T4 x2**.
   - **Internet**: Chọn **ON** (bắt buộc bật để tải thư viện).
   - **Persistence**: Chọn **Files only**.
4. Bấm **Run All**.
5. Khi hoàn tất, ở mục Output bên phải màn hình, tải về file zip tương ứng (`Task2_Baselines_Outputs.zip`, `Task2_Modular_Ablations_Outputs.zip`, `Task2_Proposed_Champion_Outputs.zip`).
