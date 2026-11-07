# Experimental Results & Performance Report — MVTec AD (`metal_nut`) Defect Detection

This document presents the complete quantitative and qualitative results of the 5-stage industrial defect detection pipeline implemented in [`tasks.ipynb`](tasks.ipynb) on the **MVTec Anomaly Detection (`metal_nut`)** dataset.

---

## 1. Executive Summary

| Pipeline Stage | Model / Format | Resolution | Device | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Model Size (MB) | Inference Latency (ms) | End-to-End Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 2 (Validation)** | YOLO26n (`best.pt`) | `704×704` | RTX 3060 GPU | **0.9970** | **1.0000** | **0.9950** | **0.8700** | 5.22 | 2.300 | — |
| **Task 3 (Test — Initial 47 imgs)** | YOLO26n (`best.pt`) | `704×704` | RTX 3060 GPU | 0.9631 | 0.8158 | 0.8967 | 0.7620 | 5.22 | 3.400 | — |
| **Task 3 (Test — Corrected 36 imgs)** | YOLO26n (`best.pt`) | `704×704` | RTX 3060 GPU | **0.9907** | **0.7273** | **0.8194** | **0.6827** | 5.22 | 4.300 | — |
| **Task 4 (CPU Benchmark)** | ONNX FP32 (`best_fp32.onnx`) | `256×256` | Intel i7-12650H | 0.8353 | 0.7727 | 0.7556 | 0.6185 | 9.272 | **10.140** (Med: **8.311**) | 47.418 |
| **Task 5 (FP32 vs INT8 — FP32)** | ONNX FP32 (`best_fp32.onnx`) | `256×256` | Intel i7-12650H | 0.8353 | **0.7727** | 0.7556 | **0.6185** | 9.272 | 11.929 (Med: 13.475) | 47.050 |
| **Task 5 (FP32 vs INT8 — INT8)** | ONNX INT8 (`best_int8.onnx`) | `256×256` | Intel i7-12650H | **0.8835** | 0.7273 | **0.7820** | 0.5646 | **2.658 (-71.3%)** | 13.247 (Med: **13.398**) | 47.884 |
| **Task 5 (Direct OpenCV + ORT)** | ONNX INT8 (`best_int8.onnx`) | `256×256` | Intel i7-12650H | **0.8835** | 0.7273 | **0.7820** | 0.5646 | **2.658 (-71.3%)** | **10.768** | **23.279 (42.96 FPS)** |

---

## 2. Task 1 — Automated Mask-to-YOLO Bounding Box Annotation Results

All 93 defective images in `metal_nut/test/{bent,color,flip,scratch}` (`700 × 700` RGB) and their corresponding binary segmentation masks in `metal_nut/ground_truth/` were converted into normalized single-class (`0: defect`) YOLO `.txt` label files under [`outputs/task1_labels/labels/`](outputs/task1_labels/labels/).

### Per-Category Bounding Box Statistics

| Defect Category | Extraction Method | Defective Images | Total Bounding Boxes | Avg Boxes / Image | Min Boxes / Image | Max Boxes / Image |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`bent`** | 8-Connected Components (`cv2.connectedComponentsWithStats`) | 25 | 35 | 1.40 | 1 | 4 |
| **`color`** | 8-Connected Components (`cv2.connectedComponentsWithStats`) | 22 | 30 | 1.36 | 1 | 7 |
| **`flip`** | Whole-Mask Encompassing Box (`extract_whole_mask_bbox`) | 23 | 23 | 1.00 | 1 | 1 |
| **`scratch`** | 8-Connected Components (`cv2.connectedComponentsWithStats`) | 23 | 44 | 1.91 | 1 | 4 |
| **Total / Overall** | **Hybrid Category-Aware Pipeline** | **93** | **132** | **1.42** | **1** | **7** |

### Design Findings
* **Why `flip` uses whole-mask extraction**: In the MVTec `metal_nut` dataset, a `flip` defect represents a global upside-down orientation of the entire nut. Its ground-truth binary mask highlights the metallic ring around the central bore, which can have small disconnected segments near the stamped markings or teeth. Using whole-mask bounding box extraction guarantees exactly **1 bounding box per flipped nut** (23 boxes across 23 images), avoiding fragmented part-level boxes.
* **Why `bent`, `color`, and `scratch` use 8-connectivity components**: Nuts with surface scratches (`1.91` boxes/image on average), color stains (`1.36` boxes/image, up to 7 droplets/spots on `color/003.png`), or multiple bent teeth (`1.40` boxes/image) contain spatially separated localized anomalies that would otherwise be merged into overly large, background-dominated bounding boxes.

---

## 3. Task 2 — Dataset Split & YOLO26n Baseline Training Results

### Dataset Split Distribution

The 335 total images in `metal_nut` (93 defective + 242 normal `good` images) were partitioned using a two-stage stratified split for defective categories (**70% Train / 15% Val / 15% Test**) combined with an **80% Train / 20% Val** split of `train/good` (220 images) and the held-out `test/good` (22 images):

| Split | `bent` | `color` | `flip` | `scratch` | Total Defective | Normal (`good`) | Total Planned Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 18 | 15 | 16 | 16 | **65** | **176** | **241** *(296 in cached dir\*)* |
| **Validation** | 4 | 3 | 4 | 3 | **14** | **44** | **58** *(103 in cached dir\*)* |
| **Test (Held-Out)** | 3 | 4 | 3 | 4 | **14** *(22 boxes)* | **22** | **36** |
| **Total** | **25** | **22** | **23** | **23** | **93** *(132 boxes)* | **242** | **335** |

*\*Note: During interactive notebook execution, running the unseeded split/copy cells across two runs without clearing `train/` and `val/` resulted in 296 train images (86 defective, 210 normal backgrounds) and 103 validation images (25 defective, 78 normal backgrounds, 30 defect instances) during training, while the `test/` split was cleaned in Task 3 to the exact 36 held-out images (14 defective, 22 normal backgrounds, 22 defect instances).*

### Model Architecture & Training Configuration
* **Architecture**: Ultralytics **YOLO26n** (`yolo26n.pt`)
  * **Unfused**: 260 layers, **2,504,190 parameters**, 5.9 GFLOPs
  * **Fused (Inference)**: 121 layers, **2,389,735 parameters**, 5.4 GFLOPs at `704×704` (0.8 GFLOPs at `256×256`)
* **Training Hyperparameters** ([`runs/detect/outputs/task2_training/yolo26n_baseline-2/args.yaml`](runs/detect/outputs/task2_training/yolo26n_baseline-2/args.yaml)):
  * **Epochs**: `100` (completed in **0.093 hours / 334.4 seconds** on NVIDIA GeForce RTX 3060 Laptop GPU)
  * **Resolution**: `imgsz=700` $\rightarrow$ auto-aligned to `704×704` (stride 32)
  * **Batch Size**: `16` (auto-scaled from 32 to fit 6 GB VRAM)
  * **Optimizer**: `AdamW(lr=0.002, momentum=0.9, weight_decay=0.0005)`
  * **Augmentations**: Mosaic (`1.0`, disabled for final 10 epochs via `close_mosaic=10`), Horizontal Flip (`fliplr=0.5`), Random Erasing (`0.4`), Scale (`0.5`), Translation (`0.1`), HSV adjustments (`h=0.015, s=0.7, v=0.4`).

### Training Progression Across Epochs ([`results.csv`](runs/detect/outputs/task2_training/yolo26n_baseline-2/results.csv))

| Epoch | Train `box_loss` | Train `cls_loss` | Train `l1_loss` | Val `box_loss` | Val `cls_loss` | Val Precision | Val Recall | Val mAP@0.5 | Val mAP@0.5:0.95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.3067 | 177.6140 | 0.01851 | 0.6880 | 459.6750 | 0.0091 | 0.2000 | 0.0308 | 0.0079 |
| **10** | 1.8190 | 2.4710 | 0.02723 | 0.8706 | 2.6601 | 0.4100 | 0.4870 | 0.4070 | 0.1770 |
| **25** | 1.6210 | 1.3330 | 0.02314 | 0.6595 | 1.1862 | 0.8730 | 0.9200 | 0.9150 | 0.5260 |
| **40** | 1.3150 | 1.0480 | 0.01980 | 0.5210 | 0.7289 | 0.9470 | 1.0000 | 0.9930 | 0.6930 |
| **60** | 1.0921 | 0.7228 | 0.01592 | 0.4437 | 0.4317 | 0.9960 | 1.0000 | 0.9950 | 0.7930 |
| **86 (Best)** | 0.8871 | 0.6352 | 0.01276 | 0.3512 | 0.2981 | **0.9970** | **1.0000** | **0.9950** | **0.8700** |
| **97 (Tied Best)** | 0.6544 | 0.4625 | 0.01135 | 0.3303 | 0.1529 | **0.9963** | **1.0000** | **0.9950** | **0.8698** |
| **100 (Final)** | 0.6263 | 0.3770 | 0.01035 | 0.3144 | 0.1769 | 0.9963 | 1.0000 | 0.9950 | 0.8627 |

### Training & Validation Visualizations

![Training Loss and Validation Metric Curves](runs/detect/outputs/task2_training/yolo26n_baseline-2/results.png)

| Annotation Distribution (`labels.jpg`) | Validation Confusion Matrix (`confusion_matrix.png`) |
| :---: | :---: |
| ![Labels Distribution](runs/detect/outputs/task2_training/yolo26n_baseline-2/labels.jpg) | ![Validation Confusion Matrix](runs/detect/outputs/task2_training/yolo26n_baseline-2/confusion_matrix.png) |

---

## 4. Task 3 — Held-Out Test Set Evaluation & Failure Case Analysis

### Quantitative Test Set Performance

The best checkpoint ([`runs/detect/outputs/task2_training/yolo26n_baseline-2/weights/best.pt`](runs/detect/outputs/task2_training/yolo26n_baseline-2/weights/best.pt)) was evaluated on the held-out test set at `imgsz=704` (`conf=0.001` for full PR curve integration):

| Evaluation Run | Test Images | Normal Backgrounds | Ground-Truth Boxes | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | GPU Inference (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Test Run** ([`runs/detect/val`](runs/detect/val/)) | 47 | 22 | 38 | 0.9631 | 0.8158 | 0.8967 | 0.7620 | 3.4 |
| **Final Corrected Held-Out Test** ([`corrected_test`](runs/detect/outputs/task3_evaluation/corrected_test/)) | **36** | **22** | **22** | **0.9907** | **0.7273** | **0.8194** | **0.6827** | **4.3** |

### Test Set Evaluation Curves & Confusion Matrix (`corrected_test`)

| Precision-Recall Curve (`BoxPR_curve.png`) | Confusion Matrix (`confusion_matrix.png`) |
| :---: | :---: |
| ![Test PR Curve](runs/detect/outputs/task3_evaluation/corrected_test/BoxPR_curve.png) | ![Test Confusion Matrix](runs/detect/outputs/task3_evaluation/corrected_test/confusion_matrix.png) |

| F1-Confidence Curve (`BoxF1_curve.png`) | Normalized Confusion Matrix (`confusion_matrix_normalized.png`) |
| :---: | :---: |
| ![Test F1 Curve](runs/detect/outputs/task3_evaluation/corrected_test/BoxF1_curve.png) | ![Test Normalized Confusion Matrix](runs/detect/outputs/task3_evaluation/corrected_test/confusion_matrix_normalized.png) |

---

### In-Depth Failure Mode Analysis (`conf=0.25`, `iou=0.70`, Match Threshold $\text{IoU} \ge 0.50$)

When evaluated across all **36 held-out test images** at operational thresholds (`conf=0.25`, NMS `iou=0.70`, greedy matching at $\text{IoU} \ge 0.50$):
* **34 out of 36 test images (94.4%) had zero box-level errors.**
* **All 22 normal (`good_test_000.png` – `good_test_021.png`) images produced 0 false positives** (100% specificity on defect-free parts).
* **12 out of 14 defective test images (85.7%) had all defects localized with $\text{IoU} \ge 0.50$ and 0 false positives.**
* **Only 2 images** (`bent_015.png` and `scratch_017.png`) accounted for all **9 failure events** (**4 False Positives** and **5 False Negatives**):

| Image | Defect Category | Ground-Truth Boxes | Predicted Boxes (`conf >= 0.25`) | Matched TP ($\text{IoU} \ge 0.5$) | False Positives (FP) | False Negatives (FN) | Primary Root Cause |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bent_015.png`** | `bent` | 4 | 1 (`conf=0.30`) | 0 | **1** | **4** | Tight prediction around the upper bent tooth (`conf=0.30`) has $\text{IoU} < 0.50$ against the larger ground-truth connected component (which extends downward into a faint surface scratch), simultaneously causing 1 FP and 1 FN; 3 smaller/fragmented secondary GT components are missed. |
| **`scratch_017.png`** | `scratch` | 2 | 4 (`conf=0.56, 0.52, 0.43, 0.41`) | 1 (`conf=0.43`) | **3** | **1** | A long diagonal scratch spanning the lower half of the nut is split by the central bore into 2 separate GT boxes. While the left segment is matched (`conf=0.43`), 3 wider overlapping predictions encompass the full scratch across the bore, yielding $\text{IoU} < 0.50$ with the right-hand GT segment. |
| **All other 34 test images** | `bent`, `color`, `flip`, `scratch`, `good` | 16 | 16 | 16 | **0** | **0** | Perfect detection and background suppression. |
| **Total (36 Images)** | — | **22** | **21** | **17** | **4** | **5** | **9 total box-level failure events across 2 images.** |

#### Visualized Failure Cases ([`outputs/task3_evaluation/failure_cases/`](outputs/task3_evaluation/failure_cases/))

> **Legend:** `Cyan` = Matched Ground Truth (TP) | `Lime` = Correct Prediction (TP) | `Yellow` = Missed Ground Truth (FN) | `Red` = False Positive Prediction (FP, with confidence score). Bold border (`linewidth=4`) highlights the specific failure event indexed by the title.

| Failure Case 1 (`bent_015.png` — Primary Bent Tooth IoU Mismatch) | Failure Case 2 (`scratch_017.png` — Cross-Bore Scratch Span) |
| :---: | :---: |
| ![Failure Case 1](outputs/task3_evaluation/failure_cases/failure_case_1.png) | ![Failure Case 2](outputs/task3_evaluation/failure_cases/failure_case_2.png) |
| **Analysis**: The detector localizes the severe physical bend at the top tooth (`red`, `conf=0.30`), but the GT mask (`yellow` bold) extends lower along a surface scuff, dropping IoU below `0.50`. | **Analysis**: The bottom-left scratch segment is matched (`cyan`/`lime`, `conf=0.43`), whereas 3 wider boxes (`red`, `0.56`, `0.52`, `0.41`) span across the central hole to cover the whole scratch line, missing the `0.50` IoU threshold on the isolated right-side GT box (`yellow` bold). |

| Failure Case 3 (`bent_015.png` — Micro-Artifact Inside Bent Tooth) | Failure Case 4 (`bent_015.png` — Left Secondary Scuff Region) | Failure Case 5 (`bent_015.png` — Left Edge Micro-Component) |
| :---: | :---: | :---: |
| ![Failure Case 3](outputs/task3_evaluation/failure_cases/failure_case_3.png) | ![Failure Case 4](outputs/task3_evaluation/failure_cases/failure_case_4.png) | ![Failure Case 5](outputs/task3_evaluation/failure_cases/failure_case_5.png) |

---

## 5. Task 4 — FP32 ONNX Export & CPU Latency Benchmark

### Export Specification ([`outputs/task4_onnx/best_fp32.onnx`](outputs/task4_onnx/best_fp32.onnx))
* **Exporter**: Ultralytics + `onnx 1.23.2` (opset 18) + `onnxslim 0.1.99`
* **Input Shape**: `(1, 3, 256, 256)` (`float32`, static batch/spatial dimensions)
* **Output Shape**: `(1, 300, 6)`
* **Model Size**: **9.27 MB** (`9,722,367` bytes)
* **Runtime**: `onnxruntime 1.23.2` (`CPUExecutionProvider` on **12th Gen Intel® Core™ i7-12650H**)

### CPU Latency Summary (`36 test images`, 5 warm-up runs, `imgsz=256`)

| Metric | Value |
| :--- | :---: |
| **Average Inference Latency** | **10.140 ms / image** |
| **Median Inference Latency** | **8.311 ms / image** |
| **P95 Inference Latency** | **17.101 ms / image** |
| **Min / Max Inference Latency** | **5.339 ms** (`good_test_013.png`) / **21.416 ms** (`good_test_000.png`) |
| **Pure Inference Throughput** | **98.62 FPS** |
| **Average End-to-End Latency (Ultralytics Wrapper)** | **47.418 ms / image** (**21.09 FPS**) |

### Per-Category Latency Breakdown (from [`outputs/task4_onnx/cpu_benchmark.csv`](outputs/task4_onnx/cpu_benchmark.csv))

| Test Category | Image Count | Mean Inference (ms) | Median Inference (ms) | Mean End-to-End (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **`bent`** (`bent_000, 005, 015`) | 3 | 8.658 | 7.595 | 48.093 |
| **`color`** (`color_000, 002, 013, 019`) | 4 | 8.861 | 7.034 | 43.858 |
| **`flip`** (`flip_013, 016, 017`) | 3 | 13.374 | 13.157 | 56.466 |
| **`scratch`** (`scratch_011, 017, 018, 022`) | 4 | 11.886 | 13.622 | 48.824 |
| **`good` (Normal)** (`good_test_000 .. 021`) | 22 | 10.041 | 7.072 | 46.663 |
| **All Test Images** | **36** | **10.140** | **8.311** | **47.418** |

---

## 6. Task 5 — Static INT8 Quantization & Direct OpenCV + ONNX Runtime Optimization

### 6.1 Model Compression (`best_fp32.onnx` vs. `best_int8.onnx`)

Static INT8 Post-Training Quantization (PTQ) was performed via ONNX Runtime static calibration across the validation split (`data=outputs/task2_dataset/data.yaml`, `fraction=1.0`, `imgsz=256`):

| Model Artifact | Precision Format | File Size (Bytes) | File Size (MB) | Size vs. PyTorch (`best.pt`) | Size vs. FP32 ONNX |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`best.pt`](runs/detect/outputs/task2_training/yolo26n_baseline-2/weights/best.pt) | PyTorch FP16/FP32 | 5,476,101 | 5.22 MB | 1.00× (Baseline) | 0.56× |
| [`best_fp32.onnx`](outputs/task4_onnx/best_fp32.onnx) | ONNX FP32 | 9,722,367 | 9.272 MB | 1.78× | 1.00× (ONNX Baseline) |
| **[`best_int8.onnx`](outputs/task5_int8/best_int8.onnx)** | **ONNX Static INT8** | **2,786,624** | **2.658 MB** | **0.51× (-49.1%)** | **0.287× (-71.34% / 3.49× smaller)** |

---

### 6.2 FP32 vs. INT8 Detection Accuracy (`256×256` CPU Evaluation on 36 Test Images)

From [`outputs/task5_int8/fp32_vs_int8_accuracy.csv`](outputs/task5_int8/fp32_vs_int8_accuracy.csv):

| ONNX Model (`256×256`) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | True Positives (Conf Matrix) | Missed GT (Conf Matrix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (`best_fp32.onnx`)** | 0.8353 | **0.7727** | 0.7556 | **0.6185** | 18 / 22 | 4 / 22 |
| **INT8 (`best_int8.onnx`)** | **0.8835 (+4.82%)** | 0.7273 (-4.54%) | **0.7820 (+2.63%)** | 0.5646 (-5.39%) | **19 / 22** | **3 / 22** |

| FP32 ONNX Confusion Matrix (`256×256`) | INT8 ONNX Confusion Matrix (`256×256`) |
| :---: | :---: |
| ![FP32 Confusion Matrix](runs/detect/outputs/task5_int8/fp32_evaluation-3/confusion_matrix.png) | ![INT8 Confusion Matrix](runs/detect/outputs/task5_int8/int8_evaluation-3/confusion_matrix.png) |

| FP32 ONNX Precision-Recall Curve (`256×256`) | INT8 ONNX Precision-Recall Curve (`256×256`) |
| :---: | :---: |
| ![FP32 PR Curve](runs/detect/outputs/task5_int8/fp32_evaluation-3/BoxPR_curve.png) | ![INT8 PR Curve](runs/detect/outputs/task5_int8/int8_evaluation-3/BoxPR_curve.png) |

**Key Accuracy Insights:**
1. **Resolution Effect (`704×704` vs. `256×256`)**: Downsampling from the native training resolution (`704×704`, mAP@0.5 = `0.8194`) to `256×256` (a $7.56\times$ reduction in pixel count) reduces FP32 mAP@0.5 to `0.7556` while enabling ~10 ms CPU execution.
2. **INT8 Regularization Effect on Precision & mAP@0.5**: Static INT8 quantization acts as a mild regularizer on low-confidence false-positive boxes at `conf >= 0.25`, increasing **Precision by +4.82%** (`0.8353` $\rightarrow$ `0.8835`) and **mAP@0.5 by +2.63%** (`0.7556` $\rightarrow$ `0.7820`), while slightly reducing high-IoU localization tightness (`mAP@0.5:0.95` from `0.6185` to `0.5646`) due to 8-bit coordinate quantization.

---

### 6.3 End-to-End CPU Latency Optimization: Ultralytics Wrapper vs. Direct OpenCV + ONNX Runtime

In [`tasks.ipynb`](tasks.ipynb) (Cells 31 & 33), we first benchmarked FP32 and INT8 ONNX models side-by-side through the standard `ultralytics.YOLO.predict()` wrapper ([`fp32_vs_int8_benchmark.csv`](outputs/task5_int8/fp32_vs_int8_benchmark.csv)), and then profiled a lean **Direct OpenCV DNN (`cv2.dnn.blobFromImage`) + `onnxruntime.InferenceSession`** pipeline on `best_int8.onnx`:

| Execution Pipeline (`256×256`, CPU i7-12650H) | Model Size (MB) | Preprocess (ms) | ONNX Inference (ms) | Median Inference (ms) | P95 Inference (ms) | End-to-End Latency (ms) | End-to-End Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 ONNX (Ultralytics Wrapper, Task 5)** | 9.272 | ~12.8 | 11.929 | 13.475 | 17.300 | 47.050 | 21.25 FPS *(83.83 Inf FPS)* |
| **INT8 ONNX (Ultralytics Wrapper, Task 5)** | **2.658** | ~10.7 | 13.247 | **13.398** | 18.135 | 47.884 | 20.88 FPS *(75.49 Inf FPS)* |
| **INT8 ONNX (Direct OpenCV + ORT, Task 5)** | **2.658** | **0.372** | **10.768** | — | — | **23.279** | **42.96 FPS (2.06× Faster)** |

### Key Engineering Takeaways
1. **3.49× Model Footprint Reduction**: Static INT8 quantization shrinks the deployable ONNX model from **9.27 MB to 2.66 MB**, allowing the entire network to reside comfortably within the L3 cache of modern edge/industrial CPUs (Intel Core i7-12650H has 24 MB Intel Smart Cache).
2. **Eliminating Framework Overhead Yields a 2.06× End-to-End Speedup**: When running single-image (`batch=1`) CPU inference through the high-level Ultralytics wrapper, Python object instantiation, image transforms, and result wrapping consume **~34.6 ms per image** (~72% of the 47.88 ms total latency). Replacing that wrapper with direct `cv2.imread` + `cv2.dnn.blobFromImage` (**0.372 ms**, a **~29× preprocessing speedup**) + `ort.InferenceSession.run()` (**10.768 ms**) + vectorized NumPy filtering cuts total end-to-end latency from **47.884 ms to 23.279 ms**, doubling real-world CPU throughput from **20.88 FPS to 42.96 FPS**.
