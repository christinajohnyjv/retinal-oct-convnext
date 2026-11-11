# An Explainable Deep Learning Framework for Retinal Disease Classification Using OCT Images with ConvNeXt-Tiny

Official implementation of the automated retinal disease classification framework utilizing **ConvNeXt-Tiny**, **Coordinate Attention (CA)**, **Focal Loss**, and **Grad-CAM** on Optical Coherence Tomography (OCT) images.

## 📌 Overview

Timely diagnosis of retinal abnormalities is vital to prevent irreversible vision loss. This framework addresses core challenges in medical OCT imaging—subtle pathological boundaries and severe class imbalance—through:
- **CLAHE & Aspect-Ratio Preserving Preprocessing**: Enhancing local retinal contrast and preventing geometric distortion.
- **ConvNeXt-Tiny Backbone**: Modern hierarchical convolutional network capturing fine-grained local and global retinal features.
- **Coordinate Attention (CA)**: Embedding spatial position and direction-aware channel attention to localize fluid accumulation and retinal thickening.
- **Class Imbalance Strategy**: Combining `WeightedRandomSampler` and `Focal Loss` ($\gamma = 2.0$) to focus learning on under-represented disease classes.
- **Stratified 5-Fold Ensemble**: Delivering robust cross-validation performance.
- **Explainable AI (Grad-CAM)**: Visualizing region-specific clinical saliency maps for diagnostic transparency.

---

## 🔬 Dataset (OCTDL)

The framework is evaluated on the **Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods (OCTDL)**, comprising 1,618 OCT images across 7 classes:
- **AMD**: Age-related Macular Degeneration (885 images)
- **NO**: Normal Retina (284 images)
- **DME**: Diabetic Macular Edema (143 images)
- **ERM**: Epiretinal Membrane (133 images)
- **RVO**: Retinal Vein Occlusion (93 images)
- **VID**: Vitreous Degeneration (58 images)
- **RAO**: Retinal Artery Occlusion (22 images)

---

## 📊 Experimental Results

Ablation study demonstrating progressive improvements across the proposed framework components on Stratified 5-Fold Cross Validation:

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline ConvNeXt-Tiny | 78.29 | 75.12 | 81.43 | 76.78 | 76.78 |
| + WeightedRandomSampler | 81.79 | 78.35 | 85.86 | 80.51 | 80.51 |
| + Focal Loss | 88.07 | 86.14 | 88.85 | 87.34 | 87.34 |
| **+ Coordinate Attention (Proposed)** | **92.68** | **91.45** | **93.02** | **92.12** | **92.12** |

---

## 📁 Repository Structure

```text
retinal-oct-convnext/
├── README.md                      # Project documentation and benchmark results
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules for checkpoints and datasets
└── notebooks/
    ├── 01_preprocessing.ipynb                 # CLAHE, aspect-ratio resize, zero-padding
    ├── 02_baseline_convnext.ipynb             # Baseline ConvNeXt-Tiny training
    ├── 03_convnext_coordinate_attention.ipynb # Proposed ConvNeXt-Tiny + CA + Focal Loss
    └── 04_gradcam_explainability.ipynb        # Grad-CAM heatmap visualization
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/retinal-oct-convnext.git
cd retinal-oct-convnext
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the notebooks in the designated numerical order:
1. `notebooks/01_preprocessing.ipynb`: Preprocess raw OCT images into standardized $224 \times 224$ tensors.
2. `notebooks/02_baseline_convnext.ipynb`: Train the baseline ConvNeXt-Tiny model.
3. `notebooks/03_convnext_coordinate_attention.ipynb`: Train with Coordinate Attention, Focal Loss, and Stratified 5-Fold CV.
4. `notebooks/04_gradcam_explainability.ipynb`: Generate Grad-CAM saliency heatmaps for model explainability.