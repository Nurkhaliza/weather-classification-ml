# Weather Classification ML App

Aplikasi Machine Learning untuk klasifikasi jenis cuaca menggunakan 5 algoritma:
- Logistic Regression
- Random Forest
- XGBoost
- Gaussian Naive Bayes
- SVM

## 📋 Fitur

- **Exploratory Data Analysis (EDA)** - Visualisasi komprehensif distribusi data
- **Data Preprocessing** - Encoding, scaling, dan cleaning
- **Multi-Algorithm Comparison** - Bandingkan 5 algoritma ML berbeda
- **Detailed Metrics** - Precision, Recall, F1-Score per class
- **Algorithm-Specific Visualizations**:
  - Logistic Regression: Feature Coefficients
  - Random Forest: Feature Importance (Gini)
  - XGBoost: Feature Importance (Gain) + Training vs Testing
  - Gaussian Naive Bayes: Prior Probabilities & Mean Values
  - SVM: Support Vectors Analysis
- **ROC Curves** - One-vs-Rest evaluation per class
- **Confusion Matrix** - Detailed prediction analysis

## 📊 Dataset

Dataset berisi data sintetis cuaca dengan fitur:
- Temperature, Humidity, Wind Speed, Precipitation
- Cloud Cover, Atmospheric Pressure, UV Index
- Season, Location, Visibility
- Target: Weather Type (Rainy, Sunny, Cloudy, Snowy)

## 👤 Penulis

**Nurkhaliza**
- NIM: B2D023021
- Tugas: Machine Learning & Sains Data | Semester 5 | UNIMUS

## 🚀 Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run weatherapp.py
```

## 📁 File Structure

```
UASML/
├── weatherapp.py                    # Main application
├── weather_classification_data.csv  # Dataset
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🛠️ Technologies

- Python 3.11+
- Streamlit - Web UI
- Scikit-learn - ML algorithms
- XGBoost - Gradient boosting
- Pandas & NumPy - Data manipulation
- Matplotlib & Seaborn - Visualization
