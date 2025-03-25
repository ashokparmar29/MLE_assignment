# Spectral Data Prediction API

## 📌 Overview
This FastAPI-based web service allows users to upload spectral data in CSV format and receive real-time predictions of vomitoxin (DON) levels using a pre-trained deep learning model.

---

## 🚀 Features
✅ Upload spectral data in CSV format
✅ Preprocess data automatically
✅ Load TensorFlow model and make predictions
✅ Return result

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ashokparmar29/MLE_assignment.git
cd MLE_assignment
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install streamlit fastapi uvicorn pandas numpy tensorflow joblib
```

### 4️⃣ Ensure You Have the Trained Model
Place your trained TensorFlow model (`dnn_model.h5`) in the project directory.

### 5️⃣ Run the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---
## Streamlit app
Run the Streamlit App using the following command:
```bash
streamlit run app.py
```

## 📡 API Usage

### **Endpoint: Upload File & Get Predictions**
**URL:** `POST http://127.0.0.1:8000/predict/`

#### **Using cURL:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_data.csv'
```

#### **Using Python Requests:**
```python
import requests

url = "http://127.0.0.1:8000/predict/"
files = {"file": open("sample_data.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 📌 Project Structure
```
├── RESTendpoints.py               # FastAPI application
├── dnn_model.h5          # Trained TensorFlow model
├── app.py             # Streamlit app
├── trainingFile.py             # Python file for training the model
├── solution.ipynb             # Python notebook file for experimentation 
├── ShortReport.md             # A short report
├── README.md             # Project documentation
```

---

## 🔥 Troubleshooting
- If `ModuleNotFoundError: No module named 'tensorflow'`, install it manually: `pip install tensorflow`
- If `OSError: Unable to open file (file signature not found)`, check the model path.
- If FastAPI does not start, ensure `uvicorn` is installed and run `pip install uvicorn`

---

## 🏆 Acknowledgments
- Built using **FastAPI** and **TensorFlow**
- Inspired by spectral analysis research for agricultural monitoring

📬 **Need Help?** Open an issue or reach out!

