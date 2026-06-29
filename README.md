# cnn-eurosat-dataset

# Project Summary: EuroSAT Satellite Image Classification

This document provides a highly structured, detailed, and AI-optimized summary of the EuroSAT image classification project contained in this workspace. It is designed to be easily parsed and understood by other AI models and developers.

---

## 1. Project Metadata (YAML)

```yaml
project_name: "EuroSAT Satellite Image Classification"
course: "Inteligencia Computacional - Trabajo 1"
institution: "Universidad Pública de Navarra (UPNA)"
framework: "PyTorch (v2.x)"
dataset: "EuroSAT (RGB version)"
num_classes: 10
classes:
  - "AnnualCrop"
  - "Forest"
  - "HerbaceousVegetation"
  - "Highway"
  - "Industrial"
  - "Pasture"
  - "PermanentCrop"
  - "Residential"
  - "River"
  - "SeaLake"
input_shape: [3, 64, 64]
key_experiments:
  - "Custom CNN from scratch (Baseline, Batch Normalization, Data Augmentation)"
  - "Transfer Learning with ResNet18 (Scratch, Feature Extraction, Partial Fine-tuning, Full Fine-tuning, Data Augmentation)"
  - "Full Dataset training with Data Augmentation"
```

---

## 2. Directory & File Structure

The project directory (`/home/kazyo/Projects/Trabajo 1`) contains the following assets:

| File Name                            | Type             | Size   | Description                                                                                                                             |
| :----------------------------------- | :--------------- | :----- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `00_setup_dataset.ipynb`             | Jupyter Notebook | 218 KB | Dataset ingestion, class distribution analysis, balanced subset creation, splitting, and dataloader setup.                              |
| `01_baseline_cnn.ipynb`              | Jupyter Notebook | 923 KB | Implements custom CNNs from scratch under three configurations (Baseline, BatchNorm, Data Augmentation), plus a full-dataset final run. |
| `02_pretrained_finetuning.ipynb`     | Jupyter Notebook | 1.2 MB | Explores transfer learning and fine-tuning using ResNet18 (ImageNet weights) under five configurations.                                 |
| `Informe_completo_Trabajo1.pdf`      | Academic PDF     | 1.6 MB | The final written report detailing the theoretical background, methodology, experimental setup, and analytical results.                 |
| `Presentacion_locutada_Trabajo1.mp4` | MP4 Video        | 142 MB | A voiced video presentation summarizing the project findings.                                                                           |
| `README.md`                          | Markdown File    | 1 B    | Empty workspace description file.                                                                                                       |
| `.gitignore`                         | Config File      | 5 B    | Ignores local temporary files (e.g. `test`).                                                                                            |

---

## 3. Data Pipeline & Preprocessing (`00_setup_dataset.ipynb`)

- **Dataset Size:** 27,000 total RGB images (64x64 pixels).
- **Subset Strategy:** To constrain compute time and evaluate model capacity under low-data regimes, a balanced subset of **1,000 images per class** was created (total 10,000 images).
- **Data Splits:**
  - **Train:** 70% (~7,000 images, actual: 6,999)
  - **Validation:** 15% (1,500 images)
  - **Test:** 15% (1,501 images)
- **Normalization:**
  - Custom CNNs: Min-max scaling to `[0, 1]` range (no ImageNet mean/std normalization).
  - ResNet18 Transfer Learning: Standard ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Data Augmentation:** Horizontal flips, vertical flips, and random rotations (15°).

---

## 4. Custom CNNs from Scratch (`01_baseline_cnn.ipynb`)

All models in this block are trained on the balanced subset (except the final run) using the `Adam` optimizer (learning rate $10^{-3}$) and `CrossEntropyLoss` for **15 epochs** with a batch size of **32**.

### Architecture: `EuroSATBaselineCNN`

- **Features Extractor:** 4 convolutional blocks.
  - _Block 1:_ Conv2d(3, 32, k=3, p=1) -> ReLU -> MaxPool2d(2)
  - _Block 2:_ Conv2d(32, 64, k=3, p=1) -> ReLU -> MaxPool2d(2)
  - _Block 3:_ Conv2d(64, 128, k=3, p=1) -> ReLU -> MaxPool2d(2)
  - _Block 4:_ Conv2d(128, 256, k=3, p=1) -> ReLU -> MaxPool2d(2)
- **Classifier:** Flatten -> Linear(4096, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 10).

### Architecture: `EuroSATCNNBatchNorm`

- Same as baseline, but inserts a `BatchNorm2d` layer immediately after each `Conv2d` layer and before the activation (Conv -> BatchNorm -> ReLU -> Pool).

### Experimental Configurations & Results:

1. **CNN Baseline:** Training from scratch without normalization/augmentations. Shows strong performance but signs of overfitting (93.2% train acc vs 87.3% test acc).
2. **CNN with Batch Normalization:** Surprisingly degrades test accuracy (drops to 84.9%). This indicates that Batch Normalization, when used without adjusting the learning rate, weight decay, or utilizing larger batch sizes, can sometimes lead to suboptimal convergence in smaller custom networks.
3. **CNN with Data Augmentation:** Effectively mitigates overfitting (train accuracy drops to 88.8% while test accuracy rises to 87.9%). Test loss drops significantly from 0.453 to 0.357.
4. **CNN with Full Dataset & Data Augmentation:** Training the custom CNN on the complete 27,000-image dataset (with augmentation) yields the best scratch model, proving that data volume is a primary driver for generalizing from-scratch architectures.

---

## 5. Transfer Learning & Fine-Tuning with ResNet18 (`02_pretrained_finetuning.ipynb`)

All models are based on the ResNet18 architecture. Trained on the balanced subset using the `Adam` optimizer (learning rate $10^{-3}$ for frozen/scratch, lower for fine-tuning) for **15 epochs** with a batch size of **32**.

### Experimental Configurations & Results:

1. **ResNet18 Scratch (Control):** Random initialization. ResNet18 has too many parameters (~11M) for a small dataset, leading to severe overfitting (Val Acc is only 67.7%, though Test Acc reached 85.9%).
2. **ResNet18 Pre-trained Congelado (Feature Extraction):** Feature extractor layers are frozen. Only the classification head (Linear(512, 10)) is trained. Achieves 91.6% test accuracy rapidly.
3. **ResNet18 Pre-trained Layer 3 & 4 Unfrozen (Partial Fine-tuning):** Unfreezes layers 3 and 4 of ResNet18. Shows a slight test accuracy improvement (92.0%).
4. **ResNet18 Pre-trained Completo (Full Fine-tuning):** Unfreezes all layers. Shows exceptional capability, achieving 96.87% test accuracy and fitting train data to 99.69%.
5. **ResNet18 Pre-trained Full Fine-tuning + Augmentation:** The best performing model on the balanced subset, achieving **97.73% test accuracy** and a very low test loss of **0.0744**. Data augmentation controls the overfitting observed in the standard fine-tuning configuration.

---

## 6. Global Performance Matrix

This table aggregates the final results across all experimental runs. Use this for direct model comparison.

| Model / Variant                      | Dataset Size    | Train Acc | Val Acc |  Test Acc  | Test Loss  |
| :----------------------------------- | :-------------- | :-------: | :-----: | :--------: | :--------: |
| **Custom CNN Baseline**              | 10,000 (subset) |  93.24%   | 87.40%  |   87.34%   |   0.4532   |
| **Custom CNN + BatchNorm**           | 10,000 (subset) |  90.78%   | 82.93%  |   84.94%   |   0.4364   |
| **Custom CNN + Augmentation**        | 10,000 (subset) |  88.78%   | 88.60%  |   87.94%   |   0.3573   |
| **Custom CNN + Augmentation (Full)** | 27,000 (full)   |  93.30%   | 90.74%  | **93.19%** | **0.2151** |
| **ResNet18 from Scratch**            | 10,000 (subset) |  84.47%   | 67.73%  |   85.88%   |   0.4061   |
| **ResNet18 Frozen (Feature Ext)**    | 10,000 (subset) |  91.94%   | 92.93%  |   91.61%   |   0.2309   |
| **ResNet18 Partial FT (L3+L4)**      | 10,000 (subset) |  91.63%   | 92.87%  |   92.01%   |   0.2355   |
| **ResNet18 Full Fine-Tuning**        | 10,000 (subset) |  99.69%   | 96.07%  |   96.87%   |   0.0985   |
| **ResNet18 Full FT + Augmentation**  | 10,000 (subset) |  98.40%   | 97.80%  | **97.73%** | **0.0744** |

---

## 7. Key Findings & Insights for an AI

1. **Overfitting on Small Data:** Complex architectures like ResNet18 trained from scratch on small datasets (1,000 images per class) exhibit severe overfitting and perform worse than simpler, custom-designed CNNs (85.88% vs 87.34% test accuracy).
2. **Transfer Learning Power:** Initializing ResNet18 with ImageNet weights and freezing the backbone (Feature Extraction) immediately yields a massive leap in accuracy (91.61%) and convergence speed.
3. **Fine-Tuning Superiority:** Unfreezing all layers of ResNet18 (Full Fine-tuning) allows the model to adjust pre-trained weights to remote sensing specifics (multispectral/RGB characteristics of satellite imagery differ from natural ImageNet images), leading to a high accuracy of 96.87%.
4. **Data Augmentation Necessity:** Data augmentation is highly effective in mitigating overfitting, reducing train-test generalization gaps, and lowering test losses across both custom CNNs and fine-tuned pre-trained models.
5. **Data Scaling Law:** The custom CNN's test accuracy jumped from 87.94% to 93.19% simply by increasing the training set size from the 10,000 subset to the full 27,000 dataset, showing that dataset size scales generalization performance.
