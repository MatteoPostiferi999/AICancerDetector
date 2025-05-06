# 🧬 AICancerDetector

## 📌 Introduction
**AICancerDetector** is a deep learning project that leverages a **Convolutional Neural Network (CNN)** to classify histopathological images of tumors. The goal is to assist in the **automatic detection and classification** of cancerous tissues, supporting early diagnosis and medical research.  
The project will also be integrated with a web-based Grad-CAM dashboard to enhance explainability.

---

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
git clone https://github.com/MatteoPostiferi999/CancerDetectAI.git
cd CancerDetectAI
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
``` 

3️⃣ Download the Dataset 
- The dataset used in this project is the [Breast Cancer Histopathological Database](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) from Kaggle.
If the dataset is not included, download it  and extract it into the datasets/ folder.

  
## 🏗 Model Training


1️⃣ Train the Model 

Run the following command to train the CNN model:
```bash
python src/train.py
```
This will load the dataset, train the CNN model, and save the trained weights in the results/ folder.


2️⃣ Analyze and Debug with Jupyter Notebook

Open the training notebook for tuning and visualization:
```bash
jupyter notebook notebooks/model_training.ipynb
```



## 🔬 Model Evaluation


1️⃣ Evaluate the Model 

```bash
python src/evaluate.py
```
2️⃣ Analyze Performance in Jupyter Notebook

```bash
jupyter notebook notebooks/model_evaluation.ipynb
```


## 🤖 Inference (Prediction)
Use the trained model to classify a new histopathological image:

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
- ✅ Accuracy: ...

- 🧠 Grad-CAM helps visualize which regions contributed to the decision.

- 📉 Confusion Matrix, ROC Curve, and classification metrics available in the evaluation notebook.




## 🚀 Future Improvements

- Add multi-class tumor classification (e.g., lung, brain, breast)
- Integrate a web-based Grad-CAM dashboard
- Hyperparameter optimization (Optuna or similar)




## 📜 License
This project is open-source under the MIT License.


## 🌟 Support
If you find this project useful, feel free to give it a ⭐ on GitHub and share your feedback!




