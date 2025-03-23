# 🧬 AICancerDetector

## 📌 Introduction
AICancerDetector is a deep learning project that uses a **Convolutional Neural Network (CNN)** to classify histopathological images of tumors. The model is trained on medical imaging datasets and aims to assist in the automatic detection and classification of cancerous tissues.

## 📂 Project Structure
- `src/` → Contains reusable Python scripts for dataset handling, model training, and evaluation.
- `notebooks/` → Jupyter Notebooks for data analysis, visualization, and model testing.
- `results/` → Stores trained models, evaluation metrics, and output visualizations.
- `datasets/` → The actual dataset is downloaded on Google Colab. This structure exists for development and path consistency.

- `requirements.txt` → Lists all dependencies required to run the project.

## 🚀 Installation
To set up the project, follow these steps:

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/MatteoPostiferi999/AICancerDetector.git
cd AICancerDetector
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
``` 

3️⃣ Download the Dataset 
- The dataset used in this project is the [Breast Cancer Histopathological Database](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) from Kaggle.
If the dataset is not included, download it  and extract it into the datasets/ folder.

  
🏗 Model Training

1️⃣ Train the Model Using the Script

Run the following command to train the CNN model:
```bash
python src/train.py
```
This script loads the dataset, trains the model, and saves the trained weights.


2️⃣ Analyze and Debug with Jupyter Notebook

For visualization, hyperparameter tuning, and debugging, open:

```bash
jupyter notebook notebooks/model_training.ipynb
```
This notebook provides detailed performance analysis and visualizations.





🔬 Model Evaluation

1️⃣ Evaluate the Model Using the Script

```bash
python src/evaluate.py
```
2️⃣ Analyze Performance in Jupyter Notebook

```bash
jupyter notebook notebooks/model_evaluation.ipynb
```

🤖 Inference (Making Predictions)

To use the trained model for prediction on a new histopathological image, run:

```bash
python src/inference.py --image_path path/to/image.jpg
```
Or use it in a Python script:

```python
from src.inference import predict
label = predict("datasets/sample_image.jpg")
print(f"Predicted Class: {label}")
```

## 📝 Results


🚀 Future Improvements


📜 License
This project is open-source under the MIT License.


🌟 If you like this project, consider giving it a ⭐ on GitHub!


