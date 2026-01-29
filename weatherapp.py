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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ====================== Konfigurasi Streamlit & Theme ======================
st.set_page_config(
    page_title="Weather Classification ML",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Color Palette
RED_TO_GOLD = ["#EE8080", "#ED9080", "#EC9F80", "#EAB380", "#E8C680", "#E6D480"]
PRIMARY_COLOR = "#E07070"
SECONDARY_COLOR = "#EAB380"
ACCENT_COLOR = "#ED9080"

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Judul Header
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}; text-align:center;'>🌦️ Weather Classification ML</h1>", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weather_classification_data.csv")
    return df

# Preprocessing function (cached globally)
@st.cache_data
def preprocess_data():
    df = load_data()
    
    # Create encoders
    le_cloud = LabelEncoder()
    le_season = LabelEncoder()
    le_location = LabelEncoder()
    le_target = LabelEncoder()
    
    # Fit encoders on original data
    df_processed = df.copy()
    df_processed["Cloud Cover"] = le_cloud.fit_transform(df_processed["Cloud Cover"])
    df_processed["Season"] = le_season.fit_transform(df_processed["Season"])
    df_processed["Location"] = le_location.fit_transform(df_processed["Location"])
    
    # Scale features
    scaler = StandardScaler()
    numerical_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", 
                      "Atmospheric Pressure", "UV Index", "Visibility (km)"]
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # Encode target
    y_encoded = le_target.fit_transform(df_processed["Weather Type"])
    X = df_processed.drop("Weather Type", axis=1)
    y = y_encoded
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return {
        'df': df,
        'df_processed': df_processed,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'le_cloud': le_cloud,
        'le_season': le_season,
        'le_location': le_location,
        'le_target': le_target,
        'scaler': scaler,
        'numerical_cols': numerical_cols,
        'X_columns': X.columns
    }

# Load preprocessed data
data = preprocess_data()
df = data['df']

# ====================== TAB NAVIGATION ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📖 About Dataset", "📊 Dashboard", "🤖 Machine Learning", "🏆 Cara Kerja Model Terbaik", "🎯 Prediction App", "📧 Contact Me"]
)

# ====================== TAB 1: ABOUT DATASET ======================
with tab1:
    st.header("📖 Tentang Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Dataset ini adalah data sintetis yang mensimulasikan data cuaca untuk tugas klasifikasi.**
        
        **🎯 Target:** Mengklasifikasikan cuaca menjadi 4 jenis:
        - **🌧️ Rainy** (Hujan)
        - **☀️ Sunny** (Cerah)
        - **☁️ Cloudy** (Berawan)
        - **❄️ Snowy** (Salju)
        """)
    
    with col2:
        st.metric("Total Records", len(df))
        st.metric("Total Features", len(df.columns))
    
    st.markdown("---")
    
    st.subheader("� Tujuan Analisis")
    
    st.markdown("""
    Proyek ini bertujuan untuk:
    
    1. **🔍 Exploratory Data Analysis (EDA)**
       - Memahami distribusi dan karakteristik data cuaca
       - Menganalisis hubungan antara fitur dengan jenis cuaca
       - Mengidentifikasi pola dan outlier dalam dataset
    
    2. **🤖 Model Development & Comparison**
       - Mengembangkan 5 algoritma machine learning yang berbeda:
         - Logistic Regression (Linear Model)
         - Random Forest (Ensemble - Bagging)
         - Gaussian Naive Bayes (Probabilistic)
         - Support Vector Machine (Non-linear)
         - Gradient Boosting (Ensemble - Boosting)
       - Membandingkan performa setiap algoritma
       - Mengidentifikasi model dengan akurasi terbaik
    
    3. **📈 Model Evaluation & Visualization**
       - Mengevaluasi performa model menggunakan metrik: Accuracy, Precision, Recall, F1-Score
       - Membuat confusion matrix untuk analisis klasifikasi per-class
       - Menghasilkan ROC curves untuk evaluasi threshold sensitivity
       - Visualisasi algorithm-specific insights (feature importance, coefficients, dll)
    
    4. **🎯 Practical Application**
       - Membuat interactive prediction app untuk klasifikasi cuaca real-time
       - Memberikan explanations tentang cara kerja setiap algoritma
       - Menyediakan insights untuk decision-making
    """)
    
    st.markdown("---")
    
    st.subheader("�📋 Deskripsi Fitur")
    
    features_desc = {
        "Temperature": "Suhu dalam °C (dari sangat dingin hingga sangat panas)",
        "Humidity": "Kelembaban (%)",
        "Wind Speed": "Kecepatan angin (km/jam)",
        "Precipitation (%)": "Presipitasi (%)",
        "Cloud Cover": "Deskripsi tutupan awan",
        "Atmospheric Pressure": "Tekanan atmosfer (hPa)",
        "UV Index": "Indeks UV (0-11)",
        "Season": "Musim saat data direkam",
        "Visibility (km)": "Jarak pandang (km)",
        "Location": "Lokasi pengambilan data",
        "Weather Type": "🎯 Target - Jenis cuaca"
    }
    
    for feature, desc in features_desc.items():
        st.write(f"**{feature}:** {desc}")
    
    st.markdown("---")
    
    st.subheader("📊 Data Preview")
    st.write("**5 Baris Pertama:**")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("**Statistik Deskriptif:**")
    st.dataframe(df.describe(), use_container_width=True)

# ====================== TAB 2: DASHBOARD ======================
with tab2:
    st.header("📊 Dashboard - Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Distribusi Weather Type")
        fig, ax = plt.subplots(figsize=(8, 5))
        weather_counts = df["Weather Type"].value_counts()
        colors = RED_TO_GOLD[:len(weather_counts)]
        ax.bar(weather_counts.index, weather_counts.values, color=colors, alpha=0.7, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel("Count", fontweight='bold')
        ax.set_title("Jumlah Data per Weather Type", fontweight='bold', color=PRIMARY_COLOR)
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Proporsi Weather Type")
        fig, ax = plt.subplots(figsize=(8, 5))
        weather_props = df["Weather Type"].value_counts(normalize=True) * 100
        colors = RED_TO_GOLD[:len(weather_props)]
        wedges, texts, autotexts = ax.pie(weather_props.values, labels=weather_props.index, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax.set_title("Proporsi Weather Type", fontweight='bold', color=PRIMARY_COLOR)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Distribusi Temperature")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["Temperature"], bins=30, color=SECONDARY_COLOR, alpha=0.7, edgecolor=PRIMARY_COLOR)
        ax.set_xlabel("Temperature (°C)", fontweight='bold')
        ax.set_ylabel("Frequency", fontweight='bold')
        ax.set_title("Distribusi Temperature", fontweight='bold', color=PRIMARY_COLOR)
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    with col2:
        st.subheader("💨 Distribusi Wind Speed")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["Wind Speed"], bins=30, color=ACCENT_COLOR, alpha=0.7, edgecolor=PRIMARY_COLOR)
        ax.set_xlabel("Wind Speed (km/h)", fontweight='bold')
        ax.set_ylabel("Frequency", fontweight='bold')
        ax.set_title("Distribusi Wind Speed", fontweight='bold', color=PRIMARY_COLOR)
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)

# ====================== TAB 3: MACHINE LEARNING ======================
with tab3:
    st.header("🤖 Machine Learning - Model Training & Evaluation")
    
    # Data Preprocessing
    st.subheader("⚙️ Data Preprocessing")
    
    le = LabelEncoder()
    df_processed = df.copy()
    df_processed["Cloud Cover"] = le.fit_transform(df_processed["Cloud Cover"])
    df_processed["Season"] = le.fit_transform(df_processed["Season"])
    df_processed["Location"] = le.fit_transform(df_processed["Location"])
    
    scaler = StandardScaler()
    numerical_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", 
                      "Atmospheric Pressure", "UV Index", "Visibility (km)"]
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df_processed["Weather Type"])
    X = df_processed.drop("Weather Type", axis=1)
    y = y_encoded
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.success(f"✅ Data preprocessing selesai!")
    st.write(f"- Training set: {len(X_train)} samples")
    st.write(f"- Testing set: {len(X_test)} samples")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🎯 Pilih Model Klasifikasi")
    
    model_option = st.selectbox(
        "Pilih satu model:",
        ("Logistic Regression", "Random Forest", "Gaussian Naive Bayes", "SVM", "Gradient Boosting")
    )
    
    # Model Initialization
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model_description = "**Logistic Regression** adalah algoritma linear yang memprediksi probabilitas kelas menggunakan fungsi sigmoid."
    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_description = "**Random Forest** menggunakan ensemble dari banyak decision trees untuk prediksi yang lebih robust."
    elif model_option == "Gaussian Naive Bayes":
        model = GaussianNB()
        model_description = "**Gaussian Naive Bayes** menggunakan probabilitas Bayes dengan asumsi fitur independen."
    elif model_option == "SVM":
        model = SVC(probability=True, random_state=42)
        model_description = "**Support Vector Machine (SVM)** mencari hyperplane optimal yang memaksimalkan margin antar kelas."
    elif model_option == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model_description = "**Gradient Boosting** membangun ensemble decision trees secara sequential untuk meminimalkan error."
    
    st.info(model_description)
    
    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluasi
    st.subheader("📈 Hasil Evaluasi")
    
    accuracy = accuracy_score(y_test, y_pred)
    
    class_labels = le_target.classes_
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.4f}")
    
    with col2:
        precision_avg = precision_score(y_test, y_pred, average='weighted')
        st.metric("📊 Precision", f"{precision_avg:.4f}")
    
    with col3:
        recall_avg = recall_score(y_test, y_pred, average='weighted')
        st.metric("🎪 Recall", f"{recall_avg:.4f}")
    
    with col4:
        f1_avg = f1_score(y_test, y_pred, average='weighted')
        st.metric("⚡ F1-Score", f"{f1_avg:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=class_labels,
                yticklabels=class_labels, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_option}', fontweight='bold', color=PRIMARY_COLOR)
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    
    # Per-Class Metrics
    st.subheader("📊 Per-Class Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Precision per Class:**")
        precision_scores = precision_score(y_test, y_pred, average=None)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_precision = RED_TO_GOLD[:len(class_labels)]
        ax.bar(class_labels, precision_scores, color=colors_precision, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    with col2:
        st.write("**Recall per Class:**")
        recall_scores = recall_score(y_test, y_pred, average=None)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_recall = RED_TO_GOLD[:len(class_labels)]
        ax.bar(class_labels, recall_scores, color=colors_recall, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel('Recall', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    with col3:
        st.write("**F1-Score per Class:**")
        f1_scores = f1_score(y_test, y_pred, average=None)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_f1 = RED_TO_GOLD[:len(class_labels)]
        ax.bar(class_labels, f1_scores, color=colors_f1, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel('F1-Score', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    # Algorithm-Specific Visualizations
    st.markdown("---")
    st.subheader("🔬 Algorithm-Specific Visualization")
    
    if model_option == "Logistic Regression":
        st.write("**Feature Coefficients:**")
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [SECONDARY_COLOR if x > 0 else ACCENT_COLOR for x in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_xlabel('Coefficient Value', fontweight='bold')
        ax.set_title('Feature Coefficients - Logistic Regression', fontweight='bold', color=PRIMARY_COLOR)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    elif model_option == "Random Forest":
        st.write("**Feature Importance (Top 10):**")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_importance = RED_TO_GOLD[:len(feature_importance)]
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_importance, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title('Top 10 Feature Importance - Random Forest', fontweight='bold', color=PRIMARY_COLOR)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    elif model_option == "Gradient Boosting":
        st.write("**Feature Importance (Top 10):**")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_importance = RED_TO_GOLD[:len(feature_importance)]
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_importance, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title('Top 10 Feature Importance - Gradient Boosting', fontweight='bold', color=PRIMARY_COLOR)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    elif model_option == "Gaussian Naive Bayes":
        st.write("**Prior Probabilities (P(Class)):**")
        class_counts = pd.Series(y_train).value_counts().sort_index()
        prior_probs = class_counts / len(y_train)
        prior_probs.index = class_labels
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_prior = RED_TO_GOLD[:len(class_labels)]
        ax.bar(class_labels, prior_probs.values, color=colors_prior, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel('Prior Probability', fontweight='bold')
        ax.set_title('Prior Probabilities - Gaussian Naive Bayes', fontweight='bold', color=PRIMARY_COLOR)
        ax.set_ylim([0, max(prior_probs.values) * 1.2])
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    elif model_option == "SVM":
        st.write("**Support Vectors per Class:**")
        support_info = pd.DataFrame({
            'Class': class_labels,
            'N Support Vectors': [model.n_support_[i] for i in range(len(class_labels))]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_sv = RED_TO_GOLD[:len(class_labels)]
        ax.bar(support_info['Class'], support_info['N Support Vectors'], color=colors_sv, alpha=0.6, edgecolor=PRIMARY_COLOR)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Support Vectors per Class - SVM', fontweight='bold', color=PRIMARY_COLOR)
        ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    # ROC Curve
    st.markdown("---")
    st.subheader("📈 ROC Curve")
    
    if hasattr(model, 'predict_proba'):
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_pred_proba = model.predict_proba(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
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
        fig.patch.set_facecolor('white')
        st.pyplot(fig)

# ====================== TAB 4: BEST MODEL COMPARISON ======================
with tab4:
    st.header("🏆 Cara Kerja Model Terbaik")
    
    st.info("Tab ini menampilkan perbandingan dan penjelasan algoritma machine learning yang digunakan.")
    
    # Data Preprocessing untuk semua model
    le = LabelEncoder()
    df_processed = df.copy()
    df_processed["Cloud Cover"] = le.fit_transform(df_processed["Cloud Cover"])
    df_processed["Season"] = le.fit_transform(df_processed["Season"])
    df_processed["Location"] = le.fit_transform(df_processed["Location"])
    
    scaler = StandardScaler()
    numerical_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", 
                      "Atmospheric Pressure", "UV Index", "Visibility (km)"]
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df_processed["Weather Type"])
    X = df_processed.drop("Weather Type", axis=1)
    y = y_encoded
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train semua model
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
        results.append({"Model": name, "Accuracy": f"{accuracy:.4f}"})
    
    # Tampilkan Hasil
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    
    st.subheader("📊 Perbandingan Akurasi Model")
    st.dataframe(results_df, use_container_width=True)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert accuracy to float for plotting
    results_for_plot = results_df.copy()
    results_for_plot["Accuracy"] = results_for_plot["Accuracy"].astype(float)
    results_for_plot = results_for_plot.sort_values("Accuracy", ascending=True)
    
    colors = RED_TO_GOLD[:len(results_for_plot)]
    ax.barh(results_for_plot["Model"], results_for_plot["Accuracy"], color=colors, alpha=0.7, edgecolor=PRIMARY_COLOR)
    ax.set_xlabel("Accuracy", fontweight='bold')
    ax.set_title("Perbandingan Akurasi Semua Model", fontweight='bold', color=PRIMARY_COLOR, fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(results_for_plot["Accuracy"]):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
    
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    
    # Best Model Recommendation
    best_model = results_df.iloc[0]
    st.success(f"🏆 **Model Terbaik:** {best_model['Model']} (Akurasi: {best_model['Accuracy']})")
    
    st.markdown("---")
    
    st.subheader("📚 Penjelasan Algoritma")
    
    algorithm_info = {
        "Logistic Regression": {
            "Deskripsi": "Algoritma linear yang menggunakan fungsi sigmoid untuk memprediksi probabilitas kelas.",
            "Kelebihan": ["Cepat", "Interpretable", "Baik untuk data linear"],
            "Kekurangan": ["Kurang baik untuk data non-linear kompleks"]
        },
        "Random Forest": {
            "Deskripsi": "Ensemble learning yang menggabungkan banyak decision trees untuk prediksi robust.",
            "Kelebihan": ["Baik untuk data kompleks", "Resistance terhadap overfitting", "Feature importance"],
            "Kekurangan": ["Lebih lambat", "Hard to interpret"]
        },
        "Gaussian Naive Bayes": {
            "Deskripsi": "Probabilistic classifier berbasis Bayes dengan asumsi fitur independen.",
            "Kelebihan": ["Cepat", "Simple", "Baik untuk small datasets"],
            "Kekurangan": ["Asumsi independensi sering tidak terpenuhi"]
        },
        "SVM": {
            "Deskripsi": "Mencari hyperplane optimal untuk memaksimalkan margin antar kelas.",
            "Kelebihan": ["Baik untuk high-dimensional data", "Versatile dengan kernel trick"],
            "Kekurangan": ["Slow untuk large datasets", "Sulit untuk interpret"]
        },
        "Gradient Boosting": {
            "Deskripsi": "Sequential ensemble yang membangun trees untuk meminimalkan error secara bertahap.",
            "Kelebihan": ["Akurasi tinggi", "Baik untuk complex patterns", "Stabil"],
            "Kekurangan": ["Lambat untuk training", "Rentan overfitting"]
        }
    }
    
    for algo_name, algo_info in algorithm_info.items():
        with st.expander(f"📖 {algo_name}"):
            st.write(f"**Deskripsi:** {algo_info['Deskripsi']}")
            st.write("**✅ Kelebihan:**")
            for kelebihan in algo_info['Kelebihan']:
                st.write(f"  - {kelebihan}")
            st.write("**❌ Kekurangan:**")
            for kekurangan in algo_info['Kekurangan']:
                st.write(f"  - {kekurangan}")

# ====================== TAB 5: PREDICTION APP ======================
with tab5:
    st.header("🎯 Prediction App")
    
    st.write("Masukkan nilai fitur untuk memprediksi jenis cuaca:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.slider("Temperature (°C)", -20.0, 50.0, 15.0)
        humidity = st.slider("Humidity (%)", 0.0, 150.0, 50.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 200.0, 10.0)
    
    with col2:
        precipitation = st.slider("Precipitation (%)", 0.0, 150.0, 20.0)
        pressure = st.slider("Atmospheric Pressure (hPa)", 900.0, 1050.0, 1013.0)
        uv_index = st.slider("UV Index", 0.0, 11.0, 5.0)
    
    with col3:
        visibility = st.slider("Visibility (km)", 1.0, 50.0, 10.0)
        
        cloud_cover_option = st.selectbox("Cloud Cover", ["Clear", "Overcast", "Partly Cloudy"])
        season_option = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
        location_option = st.selectbox("Location", ["Urban", "Coastal", "Rural"])
    
    if st.button("🔮 Predict Weather", use_container_width=True):
        # Use cached encoders and scalers
        le_cloud = data['le_cloud']
        le_season = data['le_season']
        le_location = data['le_location']
        le_target = data['le_target']
        scaler = data['scaler']
        numerical_cols = data['numerical_cols']
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Train best model (berdasarkan accuracy)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Prepare input using cached encoders
        cloud_cover_encoded = le_cloud.transform([cloud_cover_option])[0]
        season_encoded = le_season.transform([season_option])[0]
        location_encoded = le_location.transform([location_option])[0]
        
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind Speed': [wind_speed],
            'Precipitation (%)': [precipitation],
            'Cloud Cover': [cloud_cover_encoded],
            'Atmospheric Pressure': [pressure],
            'UV Index': [uv_index],
            'Season': [season_encoded],
            'Visibility (km)': [visibility],
            'Location': [location_encoded]
        })
        
        # Scale input
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Predict
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0]
        
        weather_types = le_target.classes_
        predicted_weather = weather_types[prediction]
        
        st.markdown("---")
        
        # Display prediction
        weather_emojis = {
            'Rainy': '🌧️',
            'Sunny': '☀️',
            'Cloudy': '☁️',
            'Snowy': '❄️'
        }
        
        emoji = weather_emojis.get(predicted_weather, '🌦️')
        
        st.success(f"### {emoji} Prediksi: **{predicted_weather}**")
        
        # Confidence
        st.subheader("📊 Confidence Score:")
        for i, weather in enumerate(weather_types):
            col1, col2 = st.columns([2, 8])
            with col1:
                st.write(f"**{weather}**")
            with col2:
                st.progress(confidence[i])
                st.write(f"{confidence[i]*100:.2f}%")

# ====================== TAB 6: CONTACT ME ======================
with tab6:
    st.header("📧 Contact Me")
    
    st.markdown("""
    ### Informasi Kontak
    
    **📝 Nama:** Nurkhaliza  
    **🎓 NIM:** B2D023021  
    **🏫 Institusi:** UNIMUS (Universitas Muhammadiyah Semarang)  
    **📚 Program:** Semester 5 - Machine Learning | Sains Data  
    
    ---
    
    ### Tentang Project
    
    Aplikasi Weather Classification ML ini adalah tugas dari mata kuliah **Machine Learning & Sains Data**.
    
    **Tujuan Project:**
    - Membandingkan performa 5 algoritma ML untuk klasifikasi cuaca
    - Mengimplementasikan data preprocessing dan feature engineering
    - Melakukan evaluasi model dengan berbagai metrik
    - Membuat interface interaktif dengan Streamlit
    
    **GitHub Repository:**
    [weather-classification-ml](https://github.com/Nurkhaliza/weather-classification-ml)
    
    ---
    
    ### Teknologi yang Digunakan
    
    - **Framework:** Streamlit
    - **ML Libraries:** scikit-learn
    - **Data Processing:** pandas, numpy
    - **Visualization:** matplotlib, seaborn
    - **Version Control:** Git & GitHub
    
    ---
    
    ### Terima Kasih! 🙏
    
    Semoga aplikasi ini bermanfaat untuk pembelajaran Machine Learning!
    """)
