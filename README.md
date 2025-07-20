# 🏋️‍♂️ Exercise Classifier using Smartwatch Sensor Data

A machine learning-based system that classifies **barbell exercises in real-time** using accelerometer and gyroscope data from a smartwatch. This project demonstrates the use of signal processing, feature engineering, and model training to accurately identify common strength training movements.


---

## 📊 Overview

**Goal**: Classify 5 barbell exercises using sensor data collected from a smartwatch.

### 🏋️‍♀️ Exercises Covered
- Squat
- Bench Press
- Overhead Press
- Barbell Row
- Deadlift

---

## 📈 Methodology

### 🧪 Data Collection
- Recorded **time-series sensor data** using a MetaMotion smartwatch.
- Collected data includes **3-axis accelerometer** and **gyroscope** readings.

### 🛠 Feature Engineering
- Applied:
  - **Fast Fourier Transform (FFT)** for frequency domain insights
  - **Principal Component Analysis (PCA)** for dimensionality reduction
  - **K-Means Clustering** for pattern discovery

### 🧠 Model Training
- Trained and compared multiple models:
  - **K-Nearest Neighbors (KNN)**
  - **Random Forest**
  - **Neural Network (MLPClassifier)**
- Achieved up to **98.51% classification accuracy**

---

