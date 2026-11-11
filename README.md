# An Explainable Deep Learning Framework for Retinal Disease Classification Using OCT Images with ConvNeXt-Tiny

Official implementation of the automated retinal disease classification framework utilizing **ConvNeXt-Tiny**, **Coordinate Attention (CA)**, **Focal Loss**, and **Grad-CAM** on Optical Coherence Tomography (OCT) images.

---

## 📌 Overview

Timely diagnosis of retinal abnormalities is vital to prevent irreversible vision loss. This framework addresses core challenges in medical OCT imaging—subtle pathological boundaries and severe class imbalance—through:
- **CLAHE & Aspect-Ratio Preserving Preprocessing**: Enhancing local retinal contrast and preventing geometric distortion.
- **ConvNeXt-Tiny Backbone**: Modern hierarchical convolutional network capturing fine-grained local and global retinal features.
- **Coordinate Attention (CA)**: Embedding spatial position and direction-aware channel attention to localize fluid accumulation and retinal thickening.
- **Class Imbalance Strategy**: Combining `WeightedRandomSampler` and `Focal Loss` ($\gamma = 2.0$) to focus learning on under-represented disease classes.
- **Stratified 5-Fold Ensemble**: Delivering robust cross-validation evaluation.
- **Explainable AI (Grad-CAM)**: Visualizing region-specific clinical saliency maps for diagnostic transparency.

---

## 🔬 Dataset

This framework is developed and evaluated on the **Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods (OCTDL)**.

* 🔗 **Dataset Download**: [Mendeley Data - OCTDL Dataset (doi: 10.17632/sncdhf53xc.1)](https://data.mendeley.com/datasets/sncdhf53xc/1)

The dataset contains 1,618 OCT images across 7 clinical categories:
- **AMD**: Age-related Macular Degeneration (885 images)
- **NO**: Normal Retina (284 images)
- **DME**: Diabetic Macular Edema (143 images)
- **ERM**: Epiretinal Membrane (133 images)
- **RVO**: Retinal Vein Occlusion (93 images)
- **VID**: Vitreous Degeneration (58 images)
- **RAO**: Retinal Artery Occlusion (22 images)

---

## 📁 Repository Structure

```text
retinal-oct-convnext/
├── README.md                      # Project documentation and setup guide
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules for checkpoints and datasets
└── experiments/
    ├── 01_oct_preprocessing_pipeline.ipynb
    ├── 02_convnext_baseline_training.ipynb
    ├── 03_convnext_coordinate_attention_proposed.ipynb
    └── 04_explainability_gradcam_maps.ipynb
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/retinal-oct-convnext.git
cd retinal-oct-convnext
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the notebooks in the designated numerical order:
1. **Preprocessing**: Enhances contrast with CLAHE, applies aspect-ratio padding, and formats inputs to $224 \times 224$.
2. **Baseline**: Trains standard ConvNeXt-Tiny.
3. **Proposed Model**: Integrates Coordinate Attention with Focal Loss and Stratified 5-Fold Cross Validation.
4. **Explainability**: Generates Grad-CAM activation heatmaps highlighting pathological regions.