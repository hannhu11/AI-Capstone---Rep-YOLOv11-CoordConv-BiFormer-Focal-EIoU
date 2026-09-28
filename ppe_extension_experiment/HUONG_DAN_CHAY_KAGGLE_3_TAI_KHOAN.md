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
│   TASK 2: DATA & 2 BASELINES    │   │  TASK 2: ABLATION A1 TO A5      │   │  TASK 2: PROPOSED CHAMPION (A6) │
├─────────────────────────────────┤   ├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • Phase 1: Data Pipeline,       │   │ • Phase 3: Modular Ablation     │   │ • Phase 3 & 4: Proposed         │
│   Data Cleaning, Imbalance      │   │   Studies (CHV 6-Class):        │   │   Champion A6 (Full Fusion)     │
│   Analysis (Slide 3-6)          │   │   - A1: Data Augmentation      │   │ • Quantitative Comparison Table │
│ • Phase 2: Baseline 1 (YOLO11s) │   │   - A2: CoordConv Spatial Stem  │   │   (Baselines vs A1-A6)          │
│   & Baseline 2 (YOLOv8s)        │   │   - A3: RepConv Re-param        │   │ • Failure Cases Analysis (10+)  │
│   60 Epochs tu dau              │   │   - A4: Focal-EIoU Loss         │   │ • Latency & FPS Profiling on T4 │
│ • Gap Analysis (Slide 8-9)      │   │   - A5: BiFormer Attention      │   │ • Visual Demo Predictions Grid  │
│ • Export: Task2_Baselines.zip   │   │ • Export: Task2_Ablations.zip   │   │ • Export: Task2_Champion.zip    │
└─────────────────────────────────┘   └─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 2. CHI TIET TUNG NOTEBOOK

### 2.1. TAI KHOAN KAGGLE 1
- **File Notebook**: `ppe_extension_experiment/notebooks/Task2_Account_1_DataPipeline_and_Baselines.ipynb`
- **Cau hinh Kaggle**:
  - Accelerator: **GPU T4 x2** (Dual Tesla T4)
  - Internet: **ON** (Bat buoc bat de tai thu vien va dataset)
  - Persistence: **Files only**
  - Environment: **Latest Container Image**
- **Cac buoc thao tac tren Kaggle**:
  1. Mo Kaggle Acc 1 -> Create New Notebook -> Chon File -> Upload Notebook -> Chon `Task2_Account_1_DataPipeline_and_Baselines.ipynb`.
  2. Menu ben phai: Chon Accelerator: **GPU T4 x2**, Internet: **ON**.
  3. (Tuy chon) Neu da co dataset CHV tren Kaggle, an **Add Input** -> Search `chv-dataset` -> Add. Neu chua co, notebook se **tu dong tai qua Google Drive (419 MB)** trong Cell 2.
  4. An **Run All** (Thoi gian chay uoc tinh: ~25-30 phut cho ca 2 Baseline).
- **Cac san pham thu duoc (Output Artifacts)**:
  - File zip tu dong dong goi: **`Task2_Baselines_Outputs.zip`** (tai ve 1-click chua toan bo).
  - Checkpoints: `baseline1_yolo11s_best.pt`, `baseline2_yolov8s_best.pt`.
  - Bang du lieu: `data_imbalance_analysis.csv`, `baseline_comparison.csv`, `baseline_gap_analysis.csv`.
  - Hinh anh bieu do: `class_distribution_histogram.png`, `baselines_sample_predictions.jpg`.
  - Bao cao Markdown: `TASK2_PHASE1_PHASE2_REPORT.md`.
- **Anh xa vao Slide Bao cao**:
  - **Slide 3 - 6**: Phan tich du lieu, ty le mat can bang (Imbalance Ratio), giai phap giam thieu.
  - **Slide 7 - 9**: Thiet lap 2 Baseline, so lieu thuc nghiem, so sanh voi cong bo goc cua Paper (Gap Analysis).

---

### 2.2. TAI KHOAN KAGGLE 2
- **File Notebook**: `ppe_extension_experiment/notebooks/Task2_Account_2_Modular_Ablations_A1_to_A5.ipynb`
- **Cau hinh Kaggle**:
  - Accelerator: **GPU T4 x2**
  - Internet: **ON**
  - Persistence: **Files only**
- **Cac buoc thao tac tren Kaggle**:
  1. Mo Kaggle Acc 2 -> Upload `Task2_Account_2_Modular_Ablations_A1_to_A5.ipynb`.
  2. Bat Accelerator **GPU T4 x2**, Internet **ON**.
  3. An **Run All** (Thoi gian chay uoc tinh: ~45-50 phut cho 5 thi nghiem A1 den A5, moi thi nghiem 50 epochs).
- **Cac san pham thu duoc (Output Artifacts)**:
  - File zip tu dong dong goi: **`Task2_Modular_Ablations_Outputs.zip`**.
  - Checkpoints: `a1_aug_best.pt`, `a2_coordconv_best.pt`, `a3_repconv_best.pt`, `a4_focaleiou_best.pt`, `a5_biformer_best.pt`.
  - Bang so lieu tien trinh dong gop: `modular_ablations_comparison.csv`.
  - Bieu do cot tien trinh cai thien: `modular_ablations_chart.png`.
  - Bao cao Markdown: `TASK2_PHASE3_MODULAR_ABLATIONS_REPORT.md`.
- **Anh xa vao Slide Bao cao**:
  - **Slide 10 - 11**: Tien trinh cai tien qua tung phien ban V1 -> Vn, tin hieu cai thien va ly do lua chon/loai bo tung module.
  - **Slide 13**: Bang tong hop Ablation Study modular (A1 -> A5) voi so lieu thuc chung ro rang.

---

### 2.3. TAI KHOAN KAGGLE 3
- **File Notebook**: `ppe_extension_experiment/notebooks/Task2_Account_3_Proposed_Champion_A6_and_Failure_Analysis.ipynb`
- **Cau hinh Kaggle**:
  - Accelerator: **GPU T4 x2**
  - Internet: **ON**
  - Persistence: **Files only**
- **Cac buoc thao tac tren Kaggle**:
  1. Mo Kaggle Acc 3 -> Upload `Task2_Account_3_Proposed_Champion_A6_and_Failure_Analysis.ipynb`.
  2. Bat Accelerator **GPU T4 x2**, Internet **ON**.
  3. An **Run All** (Thoi gian chay uoc tinh: ~20-25 phut).
- **Cac san pham thu duoc (Output Artifacts)**:
  - File zip tu dong dong goi: **`Task2_Proposed_Champion_Outputs.zip`**.
  - Checkpoints: `champion_a6_best.pt`, `champion_a6_fused_deploy.pt` (trong so sau khi chuyen doi switch_to_deploy san sang cho edge camera).
  - Ho so chan doan loi: `failure_cases_analysis.csv` (10 truong hop loi duoc phan loai khoa hoc kem giai phap khac phuc), `failure_cases_diagnosis_grid.png`.
  - Bang do dac phan cung: `hardware_speed_benchmark.csv` (Latency ms va FPS tren Tesla T4).
  - Bang so sanh Master: `task2_master_ablation_table.csv` (Tong hop toan bo tu Baseline 1, 2 den A1-A6).
  - Luoi anh demo du doan: `champion_sample_predictions.jpg`.
  - Bao cao Markdown: `TASK2_PHASE4_PROPOSED_CHAMPION_REPORT.md`.
- **Anh xa vao Slide Bao cao**:
  - **Slide 12**: Phan tich chi tiet cac truong hop mo hinh du doan sai (Failure Cases Analysis) - 10 vi du thuc te voi hinh anh truc quan va nguyen nhan goc re.
  - **Slide 14**: Bang so sanh tong hop tat ca cac bien the mo hinh (Toan bo chuoi A1 den A6).
  - **Slide 15**: Do dac toc do thoi gian thuc va tieu thu tai nguyen (Latency/FPS tren Tesla T4).
  - **Slide 16**: Minh hoa ket qua dau ra truc quan (Output Visual Demo) tren anh thuc te cong truong.

---

## 3. LUU Y QUAN TRONG CHO NGUOI DUNG

1. **Khong bi loi Font / Khong chua Icon bieu cam**:
   - Toan bo 3 notebook duoc lap trinh voi tieu chuan hoc thuat nghiem ngat, su dung cac the vuong chuan: `[INFO]`, `[STAGE 1]`, `[SUCCESS]`, `[EVALUATION]`, `[WARNING]`, tuyet doi khong co emoji hay icon unicode gay loi hien thi.
2. **Co che tai du lieu thong minh (Universal Dataset Loader)**:
   - Moi notebook deu ho tro tu dong tim kiem ca thu muc giai nen `CHV_dataset` lan file zip `CHV.zip`. Neu trong `/kaggle/input/` chua co, notebook se tu dong dung `gdown` tai file zip goc 419 MB tu Google Drive ma khong can thao tac thu cong.
3. **Chi can tai ve 3 file Zip sau khi chay xong**:
   - `Task2_Baselines_Outputs.zip` (tu Acc 1)
   - `Task2_Modular_Ablations_Outputs.zip` (tu Acc 2)
   - `Task2_Proposed_Champion_Outputs.zip` (tu Acc 3)
   - Khi giai nen ra, ban se co day du 100% bang so lieu CSV, hinh anh PNG 200 DPI va file bao cao Markdown de chen truc tiep vao Slide thuyet trinh Review 2!
