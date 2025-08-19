# Small Object Detection YOLO DOTA

This repository contains all code, trained model and scripts regarding a research about small object detection using different YOLO versions (v5, v8, v11, v12) applied to the **DOTA** dataset.

## Repository Structure

### 📂 article
Contains presentations and research papers related to the project.
- `Apresentação Yolo Tiny ML com Dota - v2 - Jonas.pptx.pdf` – presentation slides
- `Comparative_study_of_small_objects_detection_using_YOLO_and_DOTA_dataset.pdf` – research paper

### 📂 code
Contains Colab notebooks for data processing, model training, and evaluation.
- Main notebooks:
  - `dota_ideal_size_check_for_dataset_slicing.ipynb` – checks optimal image sizes for slicing DOTA dataset
  - `dota_slicing_algotithm.ipynb` – dataset slicing algorithm
  - `raspberry_pi_inference_yolov11_ncnn.ipynb` – YOLOv11 inference on Raspberry Pi using NCNN
  - `yolo_obb_to_hbb_conversion.ipynb` – converts oriented bounding boxes (OBB) to horizontal bounding boxes (HBB)

#### 📂 yolo-model-evaluation-scripts
Notebooks and scripts for evaluating YOLO model results and latency. 
1. `evaluate_best_yolo5_pt.ipynb` – evaluates the best YOLOv5 checkpoint  
2. `evaluate_yolo5_results_file.ipynb` – analyzes results CSV from YOLOv5  
3. `evaluate_yolov5_latency.py` – measures YOLOv5 inference latency  
4. `evaluate_yolov5_latency_cloned.py` – alternative latency evaluation script  
5. `yolo_analyze_model_results.ipynb` – generic script used to evaluate yolov8, yolov11 e yolov12 models (N and M)

#### 📂 yolo-training-scripts
Notebooks for training different YOLO versions with various image sizes, batch sizes, and optimizers.
- Example notebooks:
  - `yolo_trainings_100epochs_imgsz1024_AdamW_batch8_lr001_yolov11m_slicing_proposal.ipynb`
  - `yolo_trainings_100epochs_imgsz640_yolo5n_yolo5m_yolo8n_yolo8m_yolo11n_yolo11m_yolov12n_yolov12m.ipynb`

### 📂 yolo-training-results
Contains folders with training results for different YOLO versions, including metrics, plots, batch images, and trained weights.  
Each experiment folder includes:
- `args.yaml` – training arguments
- `results.csv` – metrics per epoch
- Plots: `PR_curve.png`, `F1_curve.png`, `confusion_matrix.png`, etc.
- Batch images: `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg`
- Subfolder `weights` containing `best.pt` and `last.pt` (trained weights)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jonasvm/small-object-detection-yolo-dota.git
cd small-object-detection-yolo-dota
