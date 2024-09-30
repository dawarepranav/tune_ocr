# OCR Model Using GOT OCR 2.0

One of the most impressive aspects of **GOT-OCR2.0** is its ability to tackle diverse Optical Character Recognition (OCR) challenges. From deciphering text in natural scenes like street signs and billboards to processing multi-page documents with intricate layouts, this AI model handles everything with precision. Originally trained for **English** and **Chinese**, this model has been **fine-tuned for Hindi** as well, expanding its versatility.

## 🔗 Model Links:
- **Fine-tuned Model (Hugging Face)**: [pranavdaware/web_ocr](https://huggingface.co/pranavdaware/web_ocr)
- **Original GOT-OCR 2.0 Repository**: [GOT-OCR 2.0 GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- **SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)**: [SWIFT GitHub](https://github.com/modelscope/ms-swift)

---

## 🚀 Features:
1. **Fine-tuned for Hindi**: In addition to English and Chinese, the model has been fine-tuned with a **Hindi OCR dataset**.
2. **Strong Validation Results**:
   - `Validation loss`: 0.41594729
   - `Accuracy`: 88.32%
   - `Gradient Norm`: 2.94309306
   - `Learning Rate`: 1.3e-07
3. **Training Data**: Fine-tuned on 5000 images from the [Hindi OCR Synthetic Line Image-Text Pair dataset](https://www.kaggle.com/datasets/prathmeshzade/hindi-ocr-synthetic-line-image-text-pair).
4. **Code Included**: The complete code for model fine-tuning is provided.

---

## 📦 Dependencies

To run this project, install the following dependencies:

```bash
pip install transformers verovio torch torchvision accelerate tiktoken gradio
```
🛠️ Usage
Follow these steps to use the fine-tuned model in your project:

Load the model and tokenizer:
```python

from transformers import AutoModel, AutoTokenizer

# Path to your fine-tuned model
model_path = 'pranavdaware/web_ocr'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True)

# Set the model to evaluation mode
model = model.eval().cuda()
```
Demo:
Try the demo in Google Colab: https://colab.research.google.com/drive/1SERRAu0tG9lLagUkOpI4eOZx2Po7z3rY?usp=sharing 

📊 Sample Data
Test the fine-tuned model with sample test images included in the repository.

🧠 Fine-Tuning Process
The fine-tuning was performed using SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning). This allows efficient model adaptation with minimal computational resources.
file : Fine_tune_code.ipynb

