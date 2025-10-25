# Multi-Modal-Rumor-Detection# Multimodal Rumor Detection (Weibo)

This project is part of a **Capstone in Applied Data Science & AI (IIT Roorkee)**.  
It implements **multimodal rumor detection** on the Weibo dataset using **deep learning models** combining **text (BERT)** and **image (ResNet)** features with different fusion strategies.

---

## 📂 Dataset
The dataset is available here:  
[Weibo Rumor Dataset (Google Drive)](https://drive.google.com/drive/folders/1kcqGBEWE7ncm_8Fn-7rq-QNfkq_fucPf?usp=sharing)

- **Text files**: rumor/non-rumor posts split into train/val/test.  
- **Image files**: corresponding rumor/non-rumor images.  
- Data merged via `post_id`.

---

## 🏗️ Models Implemented
- **Text-only**: BERT (bert-base-chinese)  
- **Image-only**: ResNet18  
- **Late Fusion**: Combines BERT & ResNet features at decision level  
- **Early Fusion**: Joint representation via Transformer encoder  
- **Attention Fusion**: Multi-head attention between image & text features (best performing)

---

## ⚙️ Training Setup
- **Optimizer**: AdamW / Adam  
- **Loss**: Weighted CrossEntropy (to handle class imbalance)  
- **Oversampling**: Applied on train set only  
- **Seed**: 42 (fixed for reproducibility)  

**Hyperparameters:**
- BERT: LR=2e-5, Dropout=0.2, Epochs=3, Batch=16  
- ResNet18: LR=1e-4, Dropout=0.3, Epochs=5, Batch=32  
- Fusion Models: LR=1e-4, Dropout=0.1–0.3, Epochs=5, Batch=8  
- Attention Heads = 8  


## 🚀 How to Run
1. Place dataset in your mounted Google Drive (already done).  
2. Update the `DATA_ROOT` variable in the code to your dataset path.  
3. Run the Python script (`.py`) or notebook in Colab with GPU enabled.  
4. Results (metrics + comparison table) will be saved as CSV.

---

## 🔑 Critical Findings
- Text-only BERT already performs very strongly.  
- **Attention Fusion** achieves best AUC and AP, confirming the benefit of cross-modal attention.  
- Early & Late Fusion underperform due to weaker feature alignment and risk of overfitting.

---

## 📌 Future Work
- Conduct **grid search** for extensive hyperparameter tuning.  
- Experiment with **ResNet50 / ViT** for images.  
- Try **RoBERTa or Chinese-specific BERT models**.  
- Data augmentation & semi-supervised learning.

---

## 📖 References
- Weibo Rumor Dataset (Google Drive)  
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019)  
- He et al., *Deep Residual Learning for Image Recognition* (ResNet, 2016)  
- Vaswani et al., *Attention is All You Need* (Transformer, 2017)  
 

---
