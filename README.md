# Skin Lesion Classification with Explainable AI

A custom PyTorch residual CNN for seven-class skin lesion classification on HAM10000. The project combines image augmentation, class-weighted training, and Grad-CAM and LIME explanation utilities.

## Dataset

HAM10000 contains 10,015 dermatoscopic images across seven lesion categories:

| Code | Category |
|---|---|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |

Images are resized to 100 × 100 and normalized. The main training script uses a stratified 80/20 training/validation split.

## Architecture and Training

| Component | Implementation |
|---|---|
| Input | 100 × 100 RGB image |
| Stem | 7 × 7 convolution and max pooling |
| Backbone | Five custom residual blocks |
| Channels | 32 → 64 → 64 → 128 → 128 |
| Head | Global average pooling → dense 512 → dense 256 → seven logits |
| Initialization | Training from scratch |
| Augmentation | Horizontal/vertical flips, rotation, and color jitter |
| Loss | Cross-entropy with square-root-scaled class weights |
| Optimizer | Adam |
| Scheduling | ReduceLROnPlateau |

## Explainability

- **Grad-CAM:** Uses class-score gradients and convolutional activations to generate spatial heatmaps.
- **LIME:** Uses perturbed image regions to estimate local feature importance.

## Results

| Metric | Value |
|---|---:|
| Validation accuracy | **75.1%** |
| Weighted F1 | **0.75** |
| Classes | **7** |

## Quick Start

Download HAM10000 separately and configure its image and metadata paths in `configs/config.py`.

```bash
python -m pip install -r requirements.txt
python train.py --epochs 30 --batch-size 32 --lr 0.001
```

After training, generate image explanations:

```bash
python explain.py --image-idx 5
```

## Outputs

Training saves a best-model checkpoint, training curves, and a confusion matrix. The explanation script generates Grad-CAM and LIME visualizations.

## Project Structure

| File | Purpose |
|---|---|
| `src/model.py` | Custom residual CNN |
| `src/dataset.py` | Dataset loading |
| `src/explainability.py` | Explanation methods |
| `train.py` | Augmentation, training, and evaluation |
| `explain.py` | Explanation generation |
| `configs/config.py` | Dataset and model settings |

## Technologies

Python, PyTorch, Torchvision, Scikit-learn, OpenCV, LIME, Pandas, NumPy, Matplotlib, and Seaborn.
