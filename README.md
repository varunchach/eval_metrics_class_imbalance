# Logistic Regression Comprehensive Demo

A comprehensive Jupyter notebook demonstrating logistic regression for binary classification, covering fundamental concepts, metrics evaluation, threshold tuning, and handling imbalanced datasets.

## 📋 Overview

This notebook provides a complete walkthrough of logistic regression, from basic implementation to advanced techniques for handling real-world challenges like imbalanced data. It's designed for beginners in Machine Learning who want to understand both the theory and practical implementation.

## 🎯 Target Audience

- Beginners in Machine Learning
- Students learning classification algorithms
- Practitioners needing a reference for logistic regression
- Anyone interested in understanding imbalanced data handling

## 📚 Topics Covered

### 1. **Basic Logistic Regression**
   - Model training and prediction
   - Probability interpretation
   - Binary classification fundamentals

### 2. **Classification Metrics**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - Classification Report
   - Understanding metric trade-offs

### 3. **Threshold Tuning**
   - Impact of threshold on precision and recall
   - Finding optimal threshold using F1-score
   - Visualizing threshold effects
   - Default vs. optimal threshold comparison

### 4. **ROC Curve and AUC**
   - Receiver Operating Characteristic curve
   - Area Under Curve (AUC) interpretation
   - Model performance evaluation

### 5. **Precision-Recall Curve**
   - PR curve for imbalanced datasets
   - AUC-PR score
   - When to use PR vs. ROC curves

### 6. **Cost Curves**
   - Cost-sensitive classification
   - Handling different misclassification costs
   - Finding threshold that minimizes total cost
   - Comparing different cost scenarios

### 7. **Handling Imbalanced Data**
   - Understanding the 90:10 imbalance problem
   - Method 1: Class weights (balanced)
   - Method 2: Oversampling/Resampling
   - Business impact of imbalanced data
   - Fraud detection case study

## 📊 Datasets Used

### 1. **Titanic Dataset** (Primary Example)
   - **Source**: Seaborn built-in dataset
   - **Size**: 891 samples
   - **Task**: Predict passenger survival
   - **Features**: Age, Fare, Sex, Class, Siblings/Spouses
   - **Class Distribution**: ~62% did not survive, ~38% survived

### 2. **Fraud Detection Dataset** (Imbalanced Data Example)
   - **Source**: Local CSV file (`base.csv`)
   - **Size**: 1,000,000 samples (sampled to 110,290 for 90:10 ratio)
   - **Task**: Predict fraudulent transactions
   - **Features**: 31 interpretable features including:
     - `income`: Customer income
     - `customer_age`: Customer age
     - `credit_risk_score`: Credit risk assessment
     - `employment_status`: Employment status
     - `housing_status`: Housing status
     - `payment_type`: Payment method
     - And more...
   - **Class Distribution**: 
     - Original: 89.7:1 (1.10% fraud rate)
     - Sampled: 9:1 (10% fraud rate) for demonstration

## 🛠️ Prerequisites

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Python Version
- Python 3.7 or higher

### Required Packages
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning algorithms and metrics
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization

## 📁 File Structure

```
ML_Demo/
├── README.md                 # This file
├── demo_ml.ipynb            # Main notebookt
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Open the Notebook**
   ```bash
   jupyter notebook demo_ml.ipynb
   ```
   Or use JupyterLab:
   ```bash
   jupyter lab demo_ml.ipynb
   ```

3. **Run Cells Sequentially**
   - The notebook is designed to be run from top to bottom
   - Each section builds upon previous concepts
   - Make sure to run all cells in order

4. **Dataset Requirements**
   - **Titanic Dataset**: Automatically loaded from Seaborn (no download needed)
   - **Fraud Detection Dataset**: Requires `base.csv` file in the Downloads folder
     - Path: `/Users/varunraste/Downloads/base.csv`
     - If the file is not available, you can skip the fraud detection section or use an alternative dataset

## 📖 Key Concepts Explained

### Classification Metrics

- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: Of predicted positives, how many are actually positive?
- **Recall**: Of actual positives, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall

### Threshold Tuning

- **Default Threshold**: 0.5 (may not be optimal)
- **Lower Threshold**: More positive predictions → Higher recall, Lower precision
- **Higher Threshold**: Fewer positive predictions → Lower recall, Higher precision
- **Optimal Threshold**: Found by maximizing F1-score or minimizing cost

### Imbalanced Data Handling

**The Problem**: With 90% legitimate and 10% fraud, a naive model predicting "always legitimate" gets 90% accuracy but catches ZERO fraud cases.

**Solutions**:
1. **Class Weights**: Automatically balance class importance
2. **Oversampling**: Duplicate minority class to balance dataset
3. **Threshold Tuning**: Adjust decision threshold to favor minority class

**Business Impact**: 
- High Recall = Catch more fraud (save money)
- Lower Precision = More false alarms (customer annoyance)
- In fraud detection, **RECALL is often more important than precision**

## 📈 Notebook Structure

1. **Introduction and Setup** (Cells 0-3)
   - Import libraries
   - Set visualization styles

2. **Titanic Dataset Analysis** (Cells 4-15)
   - Load and explore data
   - Handle missing values
   - Preprocess features
   - Split train/test sets

3. **Model Training and Evaluation** (Cells 16-28)
   - Train logistic regression
   - Calculate metrics
   - Visualize confusion matrix

4. **Threshold Tuning** (Cells 29-38)
   - Test different thresholds
   - Find optimal threshold
   - Compare default vs. optimal

5. **ROC and PR Curves** (Cells 39-44)
   - ROC curve and AUC
   - Precision-Recall curve
   - When to use each

6. **Cost Curves** (Cells 45-50)
   - Cost-sensitive classification
   - Different cost scenarios

7. **Imbalanced Data Handling** (Cells 51-69)
   - Fraud detection example
   - Standard model (baseline)
   - Class weights method
   - Oversampling method
   - Visual comparisons

## 💡 Key Insights

1. **Accuracy can be misleading** with imbalanced data
2. **Threshold selection matters** - default 0.5 may not be optimal
3. **Precision vs. Recall trade-off** - choose based on business needs
4. **ROC vs. PR curves** - PR curves are better for imbalanced data
5. **Cost considerations** - different misclassification costs require different thresholds
6. **Imbalanced data handling is critical** - especially in fraud detection, medical diagnosis, etc.

## 🔧 Customization

### Using Your Own Dataset

To use your own dataset:

1. Replace the data loading section (Cell 5 for Titanic, Cell 53 for Fraud)
2. Update feature selection and preprocessing
3. Ensure target variable is binary (0/1)
4. Adjust class distribution visualization if needed

### Adjusting Imbalance Ratio

To create different imbalance ratios:

```python
# Modify the sampling in Cell 56
n_legitimate_needed = n_fraud * 9  # Change 9 to desired ratio
```

### Changing Cost Scenarios

Modify cost values in Cell 46:

```python
cost_fp = 1  # Cost of false positive
cost_fn = 2  # Cost of false negative
```

## 📝 Notes

- The notebook uses `random_state=42` for reproducibility
- All visualizations use Seaborn's "whitegrid" style
- The fraud detection section requires the `base.csv` file
- Some cells may take longer to run (especially with large datasets)

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add more examples
- Enhance documentation

## 📄 License

This notebook is provided for educational purposes. Please ensure you have appropriate licenses for any datasets you use.

## 🔗 Additional Resources

- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Understanding ROC Curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

---

**Happy Learning! 🎓**

