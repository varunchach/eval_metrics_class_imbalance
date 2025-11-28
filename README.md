# Logistic Regression Comprehensive Demo

A comprehensive Jupyter notebook demonstrating logistic regression for binary classification, covering fundamental concepts, metrics evaluation, threshold tuning, and handling imbalanced datasets.

## 📋 Overview

This notebook provides a complete walkthrough of logistic regression, from basic implementation to advanced techniques for handling real-world challenges like imbalanced data. It covers fundamental classification metrics, threshold tuning, cost curves, and advanced evaluation techniques including lift charts, gain charts, and Kolmogorov-Smirnov (KS) statistics. It's designed for beginners in Machine Learning who want to understand both the theory and practical implementation, with special focus on business applications like fraud detection.


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

### 8. **Advanced Model Evaluation: Lift Charts, Gain Charts, and KS Statistics**
   - Decile analysis for model performance
   - Lift Charts: Measuring model performance vs. random selection
   - Gain Charts: Cumulative percentage of target captured
   - Kolmogorov-Smirnov (KS) Statistic: Maximum separation between classes
   - Combined visualizations and business insights
   - Practical applications in fraud detection and marketing

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

### Advanced Model Evaluation Metrics

In fraud detection, marketing analytics, and other business applications, we need to understand **model performance at different deciles** of the population. These metrics help answer critical business questions about resource allocation and prioritization.

#### Decile Analysis

**What is it?** Decile analysis divides the test set into 10 equal groups (deciles) based on predicted probability, from highest risk (Decile 1) to lowest risk (Decile 10).

**Why use it?** If a model is good, most positive cases (e.g., fraud) should be concentrated in the top deciles (high predicted probability). This allows businesses to prioritize investigation resources efficiently.

**Key Insight**: A well-performing model will show high concentration of positive cases in the top deciles and low concentration in the bottom deciles.

#### Lift Chart

**Definition**: Lift measures how much better our model performs compared to random selection at each decile.

**Formula**: 
```
Lift = (Fraud Rate in Decile) / (Overall Fraud Rate)
```

**Interpretation**:
- **Lift = 1.0**: Model performs same as random (no improvement)
- **Lift > 1.0**: Model is better than random (higher is better)
- **Lift < 1.0**: Model is worse than random

**Business Example**: 
- If overall fraud rate is 10% and Decile 1 has a fraud rate of 50%, then:
  - Lift = 50% / 10% = 5.0x
  - This means fraud rate in the top 10% of high-risk transactions is **5 times higher** than the average

**Visualization**: Bar chart showing lift for each decile, with a reference line at 1.0 (random performance).

**Key Insights**:
- Good models show **high lift in top deciles** (e.g., 3-5x) and **low lift in bottom deciles** (e.g., 0.1-0.5x)
- The steeper the decline from top to bottom deciles, the better the model's discrimination ability
- Lift charts help identify which segments of the population to target for maximum efficiency

**Use Cases**:
- Fraud detection: Identify which transactions to investigate first
- Marketing campaigns: Target customers most likely to respond
- Credit scoring: Prioritize high-risk applications for review

#### Gain Chart (Cumulative Gains)

**Definition**: Gain chart shows the cumulative percentage of positive cases (e.g., fraud) captured by targeting the top X% of high-risk predictions.

**Business Question**: "If we investigate only the top 20% of high-risk transactions, what percentage of all fraud cases will we catch?"

**Visualization**: 
- X-axis: Cumulative % of population (sorted by risk, from highest to lowest)
- Y-axis: Cumulative % of positive cases captured
- Three lines:
  1. **Model Performance**: Actual cumulative gain from the model
  2. **Random/Baseline**: Diagonal line (20% population = 20% fraud captured)
  3. **Perfect Model**: Would catch 100% of fraud in the top X% (where X = fraud rate)

**Interpretation**:
- **Above the diagonal**: Model is better than random
- **Closer to perfect model line**: Model is performing excellently
- **Below the diagonal**: Model is worse than random (rare, indicates a problem)

**Key Metrics from Gain Chart**:
- **Top 10% Capture**: What % of fraud is in the top 10% of high-risk transactions?
- **Top 20% Capture**: What % of fraud is in the top 20%?
- **Top 30% Capture**: What % of fraud is in the top 30%?

**Business Value**:
- Enables efficient resource allocation
- Helps set investigation thresholds (e.g., "investigate top 20% to catch 70% of fraud")
- Supports cost-benefit analysis for fraud investigation teams

**Example**: 
- If top 20% captures 70% of fraud, the business can investigate only 20% of transactions to catch 70% of fraud cases
- This is much more efficient than random investigation

#### Kolmogorov-Smirnov (KS) Statistic

**Definition**: KS statistic measures the **maximum separation** between the cumulative distribution of positive cases (fraud) and negative cases (legitimate) across all deciles.

**Formula**:
```
KS = Maximum |Cumulative % of Positive Cases - Cumulative % of Negative Cases|
```

**Range**: 0 to 1 (or 0% to 100%)

**Interpretation**:
- **KS = 0**: No separation (model is useless - cannot distinguish between classes)
- **KS = 1**: Perfect separation (model is perfect - complete separation)
- **KS > 0.5 (50%)**: Excellent model (commonly used threshold in industry)
- **KS > 0.4 (40%)**: Good model
- **KS > 0.3 (30%)**: Fair model
- **KS < 0.3 (30%)**: Poor model

**Visualization**: 
- Two curves plotted on the same graph:
  1. **Cumulative % of Positive Cases** (e.g., fraud) - typically red line
  2. **Cumulative % of Negative Cases** (e.g., legitimate) - typically blue line
- The **maximum vertical distance** between these two curves is the KS statistic
- A vertical line marks the decile where maximum separation occurs

**Key Insights**:
- Higher KS = Better model discrimination ability
- The decile where maximum separation occurs is where the model best distinguishes between classes
- KS is particularly useful in banking, credit scoring, and fraud detection industries

**Business Application**:
- **Credit Scoring**: KS > 0.4 is typically required for model approval
- **Fraud Detection**: Higher KS means better ability to separate fraud from legitimate transactions
- **Marketing**: Higher KS means better targeting of likely customers

**Why KS Matters**:
- Provides a single number to summarize model discrimination ability
- Industry-standard metric in financial services
- Helps compare different models objectively
- Indicates how well the model can separate the two classes

#### Relationship Between Metrics

These three metrics are complementary and provide different perspectives:

1. **Lift Chart**: Shows performance improvement over random at each decile
2. **Gain Chart**: Shows cumulative capture efficiency (business resource allocation)
3. **KS Statistic**: Shows overall model discrimination ability (single summary metric)

**Together, they provide**:
- **Lift**: "How much better is this decile than average?"
- **Gain**: "What % of fraud can we catch by targeting top X%?"
- **KS**: "How well does the model separate fraud from legitimate overall?"

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

7. **Imbalanced Data Handling** (Cells 51-76)
   - Fraud detection example
   - Standard model (baseline)
   - Class weights method
   - Oversampling method
   - Visual comparisons
   - ROC curve comparisons

8. **Advanced Model Evaluation** (Cells 77-96)
   - Decile analysis
   - Lift Charts
   - Gain Charts
   - KS Statistics
   - Combined visualizations
   - Comprehensive final report and dashboard

## 💡 Key Insights

1. **Accuracy can be misleading** with imbalanced data
2. **Threshold selection matters** - default 0.5 may not be optimal
3. **Precision vs. Recall trade-off** - choose based on business needs
4. **ROC vs. PR curves** - PR curves are better for imbalanced data
5. **Cost considerations** - different misclassification costs require different thresholds
6. **Imbalanced data handling is critical** - especially in fraud detection, medical diagnosis, etc.
7. **Decile analysis enables efficient resource allocation** - target top deciles to catch most positive cases
8. **Lift charts show model effectiveness** - high lift in top deciles indicates good model discrimination
9. **Gain charts guide business strategy** - answer "what % of fraud can we catch by investigating top X%?"
10. **KS statistic summarizes model quality** - KS > 0.4 is typically considered good in industry
11. **Combined metrics provide comprehensive evaluation** - use lift, gain, and KS together for complete picture

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

### Understanding Advanced Evaluation Metrics

The notebook includes comprehensive sections on:
- **Lift Charts** (Section 12.2): Shows how much better the model is than random at each decile
- **Gain Charts** (Section 12.3): Shows cumulative percentage of positive cases captured
- **KS Statistics** (Section 12.4): Measures maximum separation between classes

These metrics are particularly useful for:
- Fraud detection systems
- Marketing campaign targeting
- Credit risk assessment
- Any scenario requiring efficient resource allocation

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

### Core Concepts
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Understanding ROC Curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

### Advanced Evaluation Metrics
- [Lift Chart - Wikipedia](https://en.wikipedia.org/wiki/Lift_(data_mining))
- [Gain Chart and Lift Chart Explained](https://www.displayr.com/what-is-a-gain-chart/)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Model Evaluation Metrics for Classification](https://towardsdatascience.com/model-evaluation-metrics-for-classification-8f0d0a8c5e3)
- [Decile Analysis in Credit Risk Modeling](https://www.analyticsvidhya.com/blog/2020/11/credit-risk-modelling-using-machine-learning/)

---

**Happy Learning! 🎓**

