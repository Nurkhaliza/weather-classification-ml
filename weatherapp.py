import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====================== Konfigurasi Streamlit & Theme ======================
st.set_page_config(
    page_title="Weather Classification ML",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Color Palette (Red to Gold) - Lighter Version with 25% more intensity
RED_TO_GOLD = ["#EE8080", "#ED9080", "#EC9F80", "#EAB380", "#E8C680", "#E6D480"]
PRIMARY_COLOR = "#E07070"  # Lighter Crimson Red
SECONDARY_COLOR = "#EAB380"  # Lighter Gold
ACCENT_COLOR = "#ED9080"  # Lighter Red

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Judul Aplikasi
st.title("🌦️ Weather Classification ML App")
st.markdown("""
**Aplikasi ini melakukan klasifikasi cuaca (Rainy, Sunny, Cloudy, Snowy) menggunakan 5 algoritma Machine Learning.**
""")

# ====================== Informasi Penulis ======================
st.info("""
📝 **Informasi Penulis:**
- **Nama:** Nurkhaliza
- **NIM:** B2D023021
- **Tugas:** Machine Learning | Sains Data Semester 5 | UNIMUS

**Objective:** Membandingkan performa 5 algoritma ML untuk klasifikasi jenis cuaca dengan analisis mendalam.
""")

# Load Dataset Lokal
@st.cache_data
def load_data():
    # Gunakan relative path agar bekerja di cloud
    df = pd.read_csv("weather_classification_data.csv")
    return df

df = load_data()

# ====================== Penjelasan Dataset ======================
st.header("📚 Tentang Dataset")
st.markdown("""
**Dataset ini adalah data sintetis yang mensimulasikan data cuaca untuk tugas klasifikasi.**  
**Target:** Mengklasifikasikan cuaca menjadi 4 jenis:  
- **Rainy** (Hujan)  
- **Sunny** (Cerah)  
- **Cloudy** (Berawan)  
- **Snowy** (Salju)  

**Variabel/Fitur dalam Dataset:**  
| Variabel | Tipe Data | Deskripsi | Contoh Nilai |  
|----------|-----------|-----------|--------------|  
| **Temperature** | Numerik | Suhu dalam °C (dari sangat dingin hingga sangat panas) | -10, 25, 35 |  
| **Humidity** | Numerik | Kelembaban (%) (termasuk outlier >100%) | 40, 110 |  
| **Wind Speed** | Numerik | Kecepatan angin (km/jam) dengan nilai ekstrem | 0, 150 |  
| **Precipitation (%)** | Numerik | Presipitasi (%) (termasuk outlier) | 0, 120 |  
| **Cloud Cover** | Kategorikal | Deskripsi tutupan awan | "Clear", "Overcast" |  
| **Atmospheric Pressure** | Numerik | Tekanan atmosfer (hPa) | 980, 1013 |  
| **UV Index** | Numerik | Indeks UV (0-11) | 3, 10 |  
| **Season** | Kategorikal | Musim saat data direkam | "Winter", "Summer" |  
| **Visibility (km)** | Numerik | Jarak pandang (km) | 1, 20 |  
| **Location** | Kategorikal | Lokasi pengambilan data | "Urban", "Coastal" |  
| **Weather Type** (Target) | Kategorikal | Jenis cuaca (Rainy, Sunny, Cloudy, Snowy) | "Rainy" |  
""")

# ====================== Eksplorasi Data (EDA) ======================
st.header("📊 Exploratory Data Analysis (EDA)")

# Tampilkan 5 Data Pertama
st.write("**5 Data Pertama:**")
st.write(df.head())

# Statistik Deskriptif
st.write("**Statistik Deskriptif:**")
st.write(df.describe())

# Distribusi Target
st.write("**Distribusi Weather Type:**")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x="Weather Type", data=df, palette=RED_TO_GOLD, ax=ax)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.title("Distribusi Weather Type", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
plt.xlabel("Weather Type", fontweight='bold')
plt.ylabel("Count", fontweight='bold')
st.pyplot(fig)

# Proporsi Target
st.write("**Proporsi Weather Type (%):**")
weather_proportions = df["Weather Type"].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(8, 6))
colors_pie = RED_TO_GOLD[:len(weather_proportions)]
ax.pie(weather_proportions.values, labels=weather_proportions.index, autopct="%1.1f%%", startangle=90, colors=colors_pie, textprops={'fontweight': 'bold'})
fig.patch.set_facecolor('white')
plt.title("Proporsi Distribusi Weather Type", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
st.pyplot(fig)

# Proporsi Variabel Kategorikal
st.write("**Proporsi Variabel Kategorikal Lainnya:**")
categorical_cols = ["Cloud Cover", "Season", "Location"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('white')
for idx, col in enumerate(categorical_cols):
    col_proportions = df[col].value_counts(normalize=True) * 100
    colors_pie = RED_TO_GOLD[:len(col_proportions)]
    axes[idx].pie(col_proportions.values, labels=col_proportions.index, autopct="%1.1f%%", startangle=90, colors=colors_pie, textprops={'fontweight': 'bold'})
    axes[idx].set_title(f"Proporsi {col}", fontweight='bold', color=PRIMARY_COLOR)
plt.tight_layout()
st.pyplot(fig)

# Distribusi Variabel Numerik
st.write("**Distribusi Variabel Numerik:**")
numerical_cols_full = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", 
                       "Atmospheric Pressure", "UV Index", "Visibility (km)"]
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.patch.set_facecolor('white')
axes = axes.flatten()
for idx, col in enumerate(numerical_cols_full):
    axes[idx].hist(df[col], bins=30, color=SECONDARY_COLOR, edgecolor=PRIMARY_COLOR, alpha=0.5)
    axes[idx].set_facecolor('#f8f9fa')
    axes[idx].set_title(f"Distribusi {col}", fontweight='bold', color=PRIMARY_COLOR)
    axes[idx].set_xlabel(col, fontweight='bold')
    axes[idx].set_ylabel("Frekuensi", fontweight='bold')
for idx in range(len(numerical_cols_full), len(axes)):
    axes[idx].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

# ====================== Data Cleansing ======================
st.header("🧹 Data Cleansing")

# Cek Missing Values
st.write("**Missing Values:**")
st.write(df.isnull().sum())

# Handling Outliers (Contoh: Wind Speed)
st.write("**Boxplot Wind Speed (Sebelum Cleaning):**")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=df["Wind Speed"], ax=ax, color=SECONDARY_COLOR)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.title("Boxplot Wind Speed (Sebelum Cleaning)", fontsize=12, fontweight='bold', color=PRIMARY_COLOR)
st.pyplot(fig)

# Hapus Outlier (Wind Speed > 120 km/jam)
df = df[df["Wind Speed"] <= 120]
st.write("**Boxplot Wind Speed (Setelah Cleaning):**")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=df["Wind Speed"], ax=ax, color=SECONDARY_COLOR)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.title("Boxplot Wind Speed (Setelah Cleaning)", fontsize=12, fontweight='bold', color=PRIMARY_COLOR)
st.pyplot(fig)

# ====================== Data Preprocessing ======================
st.header("⚙️ Data Preprocessing")

# Encoding Fitur Kategorikal
st.write("**Label Encoding untuk Fitur Kategorikal:**")
le = LabelEncoder()
df["Cloud Cover"] = le.fit_transform(df["Cloud Cover"])
df["Season"] = le.fit_transform(df["Season"])
df["Location"] = le.fit_transform(df["Location"])

# Feature Scaling
st.write("**Standard Scaling untuk Fitur Numerik:**")
scaler = StandardScaler()
numerical_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", 
                  "Atmospheric Pressure", "UV Index", "Visibility (km)"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Pisahkan Features & Target
# Encode target variable (Weather Type) dari string ke numeric
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df["Weather Type"])
X = df.drop("Weather Type", axis=1)
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("**Data setelah Preprocessing:**")
st.write(X_train.head())

# ====================== Modeling & Evaluasi ======================
st.header("🤖 Modeling & Evaluasi")

# Pilih Model
model_option = st.selectbox(
    "**Pilih Model Klasifikasi:**",
    ("Logistic Regression", "Random Forest", "Gaussian Naive Bayes", "SVM", "Gradient Boosting")
)

# Inisialisasi Model
if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    model_description = """
    **Logistic Regression** adalah algoritma linear yang memprediksi probabilitas kelas menggunakan fungsi sigmoid.
    - **Kelebihan:** Cepat, interpretable, baik untuk data linear
    - **Kekurangan:** Kurang baik untuk data non-linear yang kompleks
    """
elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model_description = """
    **Random Forest** menggunakan ensemble dari banyak decision trees untuk prediksi yang lebih robust.
    - **Kelebihan:** Baik untuk data kompleks, resistance terhadap overfitting
    - **Kekurangan:** Lebih lambat, hard to interpret
    """
elif model_option == "Gaussian Naive Bayes":
    model = GaussianNB()
    model_description = """
    **Gaussian Naive Bayes** menggunakan probabilitas Bayes dengan asumsi fitur independen.
    - **Kelebihan:** Cepat, simple, baik untuk small datasets
    - **Kekurangan:** Asumsi independensi sering tidak terpenuhi di dunia nyata
    """
elif model_option == "SVM":
    model = SVC(probability=True, random_state=42)
    model_description = """
    **Support Vector Machine (SVM)** mencari hyperplane optimal yang memaksimalkan margin antar kelas.
    - **Kelebihan:** Baik untuk high-dimensional data, versatile dengan kernel trick
    - **Kekurangan:** Slow untuk large datasets, sulit untuk interpret
    """
elif model_option == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model_description = """
    **Gradient Boosting** membangun ensemble decision trees secara sequential untuk meminimalkan error.
    - **Kelebihan:** Akurasi tinggi, baik untuk complex patterns, lebih stabil dari XGBoost
    - **Kekurangan:** Lambat untuk training, rentan overfitting jika parameter tidak optimal
    """

st.write("**Model Information:**")
st.markdown(model_description)

# Latih Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)

# ====================== Step-by-Step Calculation ======================
st.write("### 📈 Perhitungan Step-by-Step")

# Info Data Split
st.write("**1️⃣ Data Split Information:**")
split_info = pd.DataFrame({
    "Informasi": ["Total Data", "Data Training (80%)", "Data Testing (20%)"],
    "Jumlah": [len(df), len(X_train), len(X_test)]
})
st.write(split_info)

# Info Model & Features
st.write("**2️⃣ Model & Feature Information:**")
model_info = pd.DataFrame({
    "Keterangan": ["Model Dipilih", "Total Features", "Classes (Target)", "Training Samples"],
    "Value": [model_option, X_train.shape[1], len(np.unique(y_train)), len(X_train)]
})
st.write(model_info)

# Metrik Detail Perhitungan
st.write("**3️⃣ Metrik Evaluasi Detail:**")
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

metrics_data = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
    "Nilai": [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"],
    "Penjelasan": [
        "Akurasi keseluruhan dari prediksi",
        "Rasio prediksi positif yang benar terhadap total prediksi positif",
        "Rasio prediksi positif yang benar terhadap total sampel positif sebenarnya",
        "Harmonic mean antara Precision dan Recall"
    ]
})
st.write(metrics_data)

# Perhitungan Manual Akurasi
st.write("**4️⃣ Manual Accuracy Calculation:**")
correct_predictions = np.sum(y_pred == y_test)
total_predictions = len(y_test)
manual_accuracy = correct_predictions / total_predictions
st.write(f"""
- **Prediksi Benar:** {correct_predictions}
- **Total Prediksi:** {total_predictions}
- **Akurasi = Prediksi Benar / Total Prediksi = {correct_predictions} / {total_predictions} = {manual_accuracy:.4f}**
""")

# Tampilkan Report Evaluasi
report = classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.write(f"**Akurasi Model:** `{accuracy:.2f}`")
st.write("**Classification Report:**")
st.text(report)

# Confusion Matrix
st.write("**Confusion Matrix:**")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
class_labels = le_target.classes_
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", xticklabels=class_labels, yticklabels=class_labels, ax=ax, cbar_kws={'label': 'Count'})
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("Actual", fontweight='bold')
plt.title("Confusion Matrix", fontsize=12, fontweight='bold', color=PRIMARY_COLOR)
st.pyplot(fig)

# ====================== Visualisasi Tambahan Per Algoritma ======================
st.write("### 📊 Visualisasi Analisis Model")

# Per-Class Metrics
st.write("**5️⃣ Per-Class Performance:**")
from sklearn.metrics import precision_recall_fscore_support
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

per_class_df = pd.DataFrame({
    'Class': class_labels,
    'Precision': [f"{p:.3f}" for p in precision_per_class],
    'Recall': [f"{r:.3f}" for r in recall_per_class],
    'F1-Score': [f"{f:.3f}" for f in f1_per_class]
})
st.write(per_class_df)

# Visualisasi Per-Class Metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('white')

colors_bar = RED_TO_GOLD[:len(class_labels)]
axes[0].bar(class_labels, precision_per_class, color=colors_bar, alpha=0.6, edgecolor=PRIMARY_COLOR)
axes[0].set_title('Precision per Class', fontweight='bold', color=PRIMARY_COLOR)
axes[0].set_ylabel('Precision', fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(class_labels, recall_per_class, color=colors_bar, alpha=0.6, edgecolor=PRIMARY_COLOR)
axes[1].set_title('Recall per Class', fontweight='bold', color=PRIMARY_COLOR)
axes[1].set_ylabel('Recall', fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(class_labels, f1_per_class, color=colors_bar, alpha=0.6, edgecolor=PRIMARY_COLOR)
axes[2].set_title('F1-Score per Class', fontweight='bold', color=PRIMARY_COLOR)
axes[2].set_ylabel('F1-Score', fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ====================== ALGORITMA-SPECIFIC VISUALIZATIONS ======================

if model_option == "Logistic Regression":
    st.write("**6️⃣ Model Coefficients (Logistic Regression Weights):**")
    st.info("Logistic Regression menggunakan koefisien linear untuk setiap fitur. Koefisien positif meningkatkan probabilitas prediksi, koefisien negatif menurunkannya.")
    
    coefficients_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]  # Untuk first class
    }).sort_values('Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    colors_coef = [RED_TO_GOLD[4] if x > 0 else RED_TO_GOLD[0] for x in coefficients_df['Coefficient']]
    ax.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], color=colors_coef, alpha=0.6, edgecolor=PRIMARY_COLOR)
    ax.set_xlabel('Coefficient Value', fontweight='bold')
    ax.set_title('Feature Coefficients - Logistic Regression', fontweight='bold', color=PRIMARY_COLOR)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    st.write(coefficients_df)

elif model_option == "Random Forest":
    st.write("**6️⃣ Feature Importance (Random Forest):**")
    st.info("Random Forest menghitung feature importance berdasarkan seberapa banyak fitur mengurangi impurity (Gini) di semua trees.")
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    colors_importance = RED_TO_GOLD[:len(feature_importance)]
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_importance, alpha=0.6, edgecolor=PRIMARY_COLOR)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Top 10 Feature Importance - Random Forest', fontweight='bold', color=PRIMARY_COLOR)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    st.write("**Top Features:**")
    st.write(feature_importance)

elif model_option == "Gaussian Naive Bayes":
    st.write("**6️⃣ Model Parameters (Gaussian Naive Bayes):**")
    st.info("Gaussian Naive Bayes menggunakan prior probability dan conditional probability berdasarkan distribusi Gaussian.")
    
    # Prior probabilities
    prior_probs = pd.Series(model.class_prior_, index=class_labels)
    
    st.write("**Prior Probabilities (P(Class)):**")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    colors_prior = RED_TO_GOLD[:len(class_labels)]
    ax.bar(class_labels, prior_probs.values, color=colors_prior, alpha=0.6, edgecolor=PRIMARY_COLOR)
    ax.set_ylabel('Prior Probability', fontweight='bold')
    ax.set_title('Prior Probabilities - Gaussian Naive Bayes', fontweight='bold', color=PRIMARY_COLOR)
    ax.set_ylim([0, max(prior_probs.values) * 1.2])
    for i, v in enumerate(prior_probs.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.write(prior_probs)
    
    # Theta (mean) and Var (variance)
    st.write("**Mean Values per Feature (First 5 features):**")
    theta_df = pd.DataFrame(model.theta_[:, :5], columns=X_train.columns[:5], index=class_labels)
    st.write(theta_df)

elif model_option == "SVM":
    st.write("**6️⃣ SVM Decision Function Coefficients:**")
    st.info("SVM menggunakan Support Vectors untuk membuat decision boundary. Kami menampilkan koefisien decision function sebagai proxy.")
    
    # Get decision function coefficients (for dual representation)
    support_info = pd.DataFrame({
        'Class': class_labels,
        'N Support Vectors': [model.n_support_[i] for i in range(len(class_labels))]
    })
    
    st.write("**Number of Support Vectors per Class:**")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    colors_sv = RED_TO_GOLD[:len(class_labels)]
    ax.bar(support_info['Class'], support_info['N Support Vectors'], color=colors_sv, alpha=0.6, edgecolor=PRIMARY_COLOR)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Support Vectors per Class - SVM', fontweight='bold', color=PRIMARY_COLOR)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(support_info['N Support Vectors']):
        ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)
    
    st.write(support_info)
    st.write(f"**Total Support Vectors:** {len(model.support_)}")
    st.write(f"**Percentage of Training Data:** {len(model.support_) / len(X_train) * 100:.1f}%")

elif model_option == "Gradient Boosting":
    st.write("**6️⃣ Feature Importance (Gradient Boosting):**")
    st.info("Gradient Boosting menampilkan feature importance berdasarkan kontribusi setiap fitur terhadap pengurangan loss di semua trees.")
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    colors_importance = RED_TO_GOLD[:len(feature_importance)]
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_importance, alpha=0.6, edgecolor=PRIMARY_COLOR)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title('Top 10 Feature Importance - Gradient Boosting', fontweight='bold', color=PRIMARY_COLOR)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    st.write("**Top Features:**")
    st.write(feature_importance)

# ROC Curve (untuk semua algoritma)
st.write("**7️⃣ ROC Curve (One-vs-Rest):**")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    colors_roc = RED_TO_GOLD[:len(class_labels)]
    for i, class_name in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', color=colors_roc[i], linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'ROC Curve - {model_option}', fontweight='bold', color=PRIMARY_COLOR)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    st.pyplot(fig)
else:
    st.warning(f"{model_option} tidak support predict_proba(), ROC Curve tidak dapat ditampilkan.")

# ====================== Rekomendasi Model Terbaik ======================
st.header("🏆 Rekomendasi Model Terbaik")

# Bandingkan Semua Model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": accuracy})

# Tampilkan Hasil
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df["Accuracy"] = results_df["Accuracy"].round(4)  # Ensure all are floats
st.write("**Perbandingan Akurasi Model:**")
st.write(results_df)

# Visualisasi Perbandingan
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette=RED_TO_GOLD[:len(results_df)], ax=ax, hue="Model", legend=False)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.title("Perbandingan Akurasi Model", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
plt.xlabel("Accuracy", fontweight='bold')
plt.ylabel("Model", fontweight='bold')
st.pyplot(fig)

# Rekomendasi Model Terbaik
best_model = results_df.iloc[0]
st.success(f"**Rekomendasi Model Terbaik:** `{best_model['Model']}` (Akurasi: `{best_model['Accuracy']:.4f}`)")