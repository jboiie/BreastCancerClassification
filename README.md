
# 🧠 Breast Cancer Prediction - With a Touch of PyTorch

Welcome to my machine learning project that tries to do something very real: predict breast cancer (malignant or benign) using a neural network powered by PyTorch and a slick Streamlit web interface.

---

## 🧬 What This Project Does

This project takes in 30 real-valued medical features (think radius, texture, smoothness, etc.) from the **Wisconsin Breast Cancer dataset** and predicts whether the tumor is **malignant** or **benign**.

Models used:
- 🧠 Custom Neural Network (PyTorch)
- 📊 Logistic Regression (sklearn - used for comparison)

---

## 🛠️ Tools & Tech Stack

- Python 3.x  
- PyTorch  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit (for the front-end interface)  
- Matplotlib (optional, for visualizations)

---

## 🖥️ How to Run This Locally

1. **Clone the repo**  
```bash
git clone https://github.com/jboiie/BreastCancerClassification.git
cd BreastCancerClassification
````

2. **Install dependencies**
   Make sure to use a virtual environment (recommended)

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📂 File Structure

```
├── app.py                      # Streamlit app
├── cancer_model.pth           # Trained PyTorch model
├── Cancer_Data.csv            # Raw dataset
├── Cancer_Data_Cleaned.csv    # Cleaned dataset used for training
├── Breast_Cancer_Prediction_test.csv  # Sample prediction test file
├── README.md
```

---

## ⚙️ Features

* Input data manually via sliders on the Streamlit UI
* Upload CSVs with multiple patient records for batch prediction
* Model is already trained — just plug and play

---

## 🙌 Credits

Dataset: [UCI ML Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
Built with way too many print statements and coffee.

---

## 📫 Contact

Built by [@jboiie](https://github.com/jboiie) — feel free to reach out or contribute!
