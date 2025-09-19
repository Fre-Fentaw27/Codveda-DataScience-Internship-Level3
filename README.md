# Codveda-DataScience-Internship-Level3

Time Series Analysis, Natural Language Processing(NLP)-Text Classification, Neural Networks with TensorFlow/Keras

## 📌 Overview

This repository contains my solutions for the Level 3 Data Science Internship tasks, covering three fundamental machine learning areas: Time Series Analysis, Natural Language
Processing(NLP)-Text Classification, and Neural Networks with TensorFlow/Keras. Each task demonstrates different aspects of data science workflow from data preprocessing to model evaluation.

## 📂 Project Structure

```bash
Codveda-DataScience-Internship-Level3/
│
├── data/
│   ├── raw/
│   │   ├── Sentiment dataset.csv                # Task 1: TimeSeries and Task 2: NLP Text Data
│   │   ├── churn-bigml-80.csv                   # Task 3: Churn Training Data
│   │   └── churn-bigml-20.csv                   # Task 3: Churn Test Data
│   └── processed/
│       ├── cleaned_sentiment.csv                # Task 1: Processed Data
│       ├── cleaned_sentiment_task2.csv          # Task 2: Cleaned Text Data
│       ├── features/
│       │   ├── tfidf_features.pkl               # Task 2: TF-IDF Features
│       │   └── tfidf_vectorizer.pkl             # Task 2: TF-IDF Vectorizer
│       └── churn_processed.pkl                  # Task 3: Processed Churn Data
│
├── src/
│   ├── Task1_TimeSeries/
│   │   ├── data_processing.py                   # Time Series Data Processing
│   │   ├── time_series_analysis.ipynb           # Time Series Analysis and Forecasting Models
│   ├── Task2_NLP/
│   │   ├── nlp_exploration
│   │   ├── nlp_data_processing.py               # Text Preprocessing
│   │   ├── nlp_feature_extraction.py            # TF-IDF Feature Extraction
│   │   ├── nlp_model_training.py                # Model Training
│   │   ├── nlp_evaluation.py                    # Model Evaluation
│   │   └── predict_sentiment.py                 # Sentiment Prediction
│   │
│   └── Task3_NeuralNetworks/
│       ├── churn_data_processing.py             # Churn Data Processing
│       ├── churn_nn_model.py                    # Neural Network Model
│       ├── churn_hyperparameter_tuning.py       # Hyperparameter Tuning
│       ├── churn_evaluation.py                  # Model Evaluation
│
├── models/
│   ├── Task1_TimeSeries/
│   │   ├── arima_model.pkl                      # ARIMA Model
│   │   ├── sarima_model.pkl                     # SARIMA Model
│   │   └── prophet_model.pkl                    # Prophet Model
│   │
│   ├── Task2_NLP/
│   │   ├── best_model.pkl                       # Best NLP Model
│   │   ├── model_comparison_results.csv         # Model Comparison
│   │   └── model_info.pkl                       # Model Information
│   │
│   └── Task3_NeuralNetworks/
│       ├── best_churn_model.h5                  # Best Neural Network
│       ├── tuned_churn_model.h5
│
├── outputs/
│   ├── Task1_TimeSeries/
│   │   ├── decomposition_plot.png               # Time Series Decomposition
│   │   ├── forecast_results.png                 # Forecast Visualization
│   │   ├── acf_pacf_plots.png                   # ACF/PACF Analysis
│   │   └── model_evaluation.txt                 # Evaluation Metrics
│   │
│   ├── Task2_NLP/
│   │   ├── class_distribution.png               # Sentiment Distribution
│   │   ├── confusion_matrix.png                 # Confusion Matrix
│   │   ├── feature_importance.png               # Feature Importance
│   │   ├── model_performance.png                # Model Comparison
│   │   ├── classification_report.txt            # Detailed Report
│   │   └── results_summary.txt                  # Results Summary
│   │
│   └── Task3_NeuralNetworks/
│       ├── accuracy_curve.png                   # Accuracy Curve
│       ├── loss_curve.png                       # Loss Curve
│       ├── confusion_matrix.png                 # Confusion Matrix
│       ├── feature_importance.png               # Feature Importance
│       ├── churn_distribution.png               # Churn Distribution
│       ├── correlation_heatmap.png              # Correlation Heatmap
│       └── model_performance.txt                # Performance Metrics
│       └──best_model_performance
│       └──roc_curve.png
│       └──comprehensive_evaluation.txt
│       └──hyperparameter_tuning_analysis.png
│       └──hyperparameter_tuning_results.csv
│       └──training_history.png
│       └──prediction_distribution.png
│       └──precision_recall_curve.png
│       └──
├── requirements.txt                             # Python Dependencies
├── .gitignore                                   # Git Ignore File
└── README.md                                    # Project Documentation
```

## 🚀 Project Tasks

## Task 1: Time Series Analysis

**Description**: Analyze and model time-series data to forecast future values

**Objectives Achieved**:

✅ Decompose time series into trend, seasonality, and residuals

✅ Implement ARIMA, SARIMA, and Prophet models

✅ Evaluate models using MAE, RMSE, and MAPE

## Task 2: Natural Language Processing (NLP) - Text Classification

**Description**: Classify text data into sentiment categories
**Objectives Achieved**:

✅ Preprocess text data (tokenization, stopwords removal, lemmatization)

✅ Convert text to numerical representation using TF-IDF

✅ Train classification models (Naive Bayes, Logistic Regression, SVM, Random Forest)

✅ Evaluate using precision, recall, and F1-score

## Task 3: Neural Networks with TensorFlow/Keras

**Description**: Build and train neural network for churn prediction

**Objectives Achieved**:

✅ Preprocess structured churn prediction data

✅ Design neural network architecture

✅ Train model using backpropagation

✅ Tune hyperparameters (learning rate, batch size)

✅ Evaluate using accuracy and loss curves

## 🛠️ Setup & Installation

1.  **Clone the repository** (if applicable) or ensure you have the project structure locally.
2.  **Navigate to the project root directory** in your terminal.
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    ```
4.  **Activate the virtual environment**:
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
5.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

1. Task 1: Time Series Analysis:

```bash
cd src/Task1_TimeSeries/
# run individually:
python src/Task1_TimeSeries/data_processing.py
python src/Task1_TimeSeries/time_series_analysis.ipynb
```

2. Task 2: Natural Language Processing (NLP) - Text Classification

```bash
cd src/Task2_NLP/
# run individually:
python nlp_exploration.ipynb
python nlp_data_processing.py
python nlp_feature_extraction.py
python nlp_model_training.py
python nlp_evaluation.py
python predict_sentiment.py
```

3. Task 3: Neural Networks with TensorFlow/Keras:

```bash
cd src/Task3_NeuralNetworks/
# run individually:
python churn_data_processing.py
python churn_nn_model.py
python churn_hyperparameter_tuning.py
python churn_evaluation.py
```

## 📊 Exploring Results

**Visualizations**: Check outputs/task\_\*/ for all generated plots and analysis visualizations

**Processed Data**: Available in data/processed/ for each task

**Trained Models**: Stored in models/task\_\*/ including optimized versions

**Evaluation Reports**: Comprehensive performance metrics and classification reports in each task's output folder

## 🔍 Results Overview

**Task 1**: Time series decomposition, forecasting results, and model comparison metrics

**Task 2**: Text classification performance, sentiment analysis results, and feature importance

**Task 3**: Neural network training history, churn prediction metrics, and hyperparameter optimization results

## 🔧 Technologies Used

**Data Processing**: pandas, numpy

**Visualization**: matplotlib, seaborn

**Time Series**: statsmodels, prophet

**NLP**: nltk, scikit-learn

**Neural Networks**: TensorFlow, Keras

**Model Evaluation**: scikit-learn metrics
