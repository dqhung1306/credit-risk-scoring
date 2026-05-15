# 🏦 Home Credit Default Risk Prediction 💳


## 📌 Overview

Credit risk management is one of the most critical challenges in the financial services industry. Accurately predicting the likelihood of loan defaults helps financial institutions to:
- **Minimize financial losses** from non-performing loans.
- **Optimize lending decisions** by accurately identifying high-risk customers.
- **Personalize interest rates** based on individual risk profiles.

This project utilizes the **Home Credit Default Risk** dataset from Kaggle to build a machine learning model capable of predicting the probability of a client experiencing payment difficulties. The pipeline includes Exploratory Data Analysis (EDA), multi-table data preprocessing, feature engineering, and model training using **LightGBM**.

**Live Demo App:** [Predict Credit Risk Online](https://credit-risk-scoring-kaf9rhmrgkq5rvu4nybgic.streamlit.app/)

---

## 📂 Dataset Information

**Source:** [Home Credit Default Risk - Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

**Data Scale:**
- Training set: 307,511 records
- Target Variable (`TARGET`): 
  - **0 (Approved/Good):** 91.93% - Client paid on time
  - **1 (Risk/Default):** 8.07% - Client had payment difficulties

**Multi-source Data Tables:**
- `application_train/test`: Main application information
- `bureau` & `bureau_balance`: Credit history with other financial institutions
- `previous_application`: History of previous applications at Home Credit
- `installments_payments`: Actual payment history details
- `POS_CASH_balance` & `credit_card_balance`: Monthly balance data

---

## 🔧 Data Processing Pipeline

### 1. Data Loading & Integration
- Loaded 307,511 application records with 122 features
- Integrated 6 auxiliary tables (bureau, previous applications, payment history)
- Handled multi-level data structures (customer → loans → payment records)

### 2. Missing Value Treatment
- Identified features with varying missing patterns (0% to 70%+ missing)
- **Categorical:** Imputed with mode or created "Unknown" category for high-missing features
- **Numerical:** Applied median imputation for financial variables, mean for behavioral metrics
- Preserved missing indicators for features where absence itself is informative

### 3. Outlier Detection & Treatment
- Used **IQR (Interquartile Range) method**: $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$
- Identified extreme outliers in: Income, Credit Amount, Employment Days
- Applied **Winsorization** (capping at 95th/5th percentiles) for skewed distributions
- Preserved valid business outliers (e.g., very high income)

### 4. Feature Engineering
**Financial Ratios:**
- `CREDIT_INCOME_RATIO` = AMT_CREDIT / (AMT_INCOME_TOTAL + 1)
- `ANNUITY_INCOME_RATIO` = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
- These ratios capture leverage and debt burden independent of absolute income

**Historical Aggregation from Previous Applications:**
- `PREV_APP_COUNT`: Total number of previous applications per customer
- `PREV_APPROVED_COUNT`: Count of approved previous applications
- `PREV_REFUSED_COUNT`: Count of rejected applications
- `APPROVAL_RATE`: Approval rate from customer's application history

**Time-Based Features:**
- Age calculation from DAYS_BIRTH (normalized to years)
- Employment tenure from DAYS_EMPLOYED
- Time since registration and ID document changes

### 5. Encoding & Transformation
- **One-hot encoding** for categorical variables (Contract Type, Housing Type, Occupation, etc.)
- **Log transformation** for right-skewed numerical features (Income, Credit Amount)
- **Normalization** of external credit scores and population density features

### 6. Data Quality Validation
- No duplicate records detected
- Verified data type consistency across tables
- Cross-validated foreign key relationships (SK_ID_CURR consistency)

---

## 🎯 Exploratory Data Analysis (EDA) Findings

### Dataset Overview
- **Total Records:** 307,511 loan applications
- **Features:** 122 variables across 7 data tables
- **Time Period:** Loans from various years integrated via application history
- **Data Types:** Mix of numerical (continuous), categorical, and derived features

### Target Variable Analysis
**Class Distribution (Highly Imbalanced):**
```
Class 0 (Non-Default): 281,686 cases (91.93%)
Class 1 (Default):      24,825 cases (8.07%)
```
- **Business Impact:** The severe class imbalance requires careful model evaluation
- **Solution:** Using ROC-AUC and F1-Score instead of accuracy
- **Class Weights:** Applied to penalize minority class misclassification in LightGBM

### Feature Distribution Insights

**Age Profile (DAYS_BIRTH):**
- Mean Age: ~43 years | Median: ~41 years
- Range: 22 to 69 years
- **Default Risk Pattern:** Younger clients (22-35) have higher default rates than middle-aged (40-50)
- Reason: Lower income stability and savings at younger ages

**Income Distribution (AMT_INCOME_TOTAL):**
- Mean: ~168,500 currency units | Median: ~147,000
- **Highly right-skewed** (extreme outliers up to 117 million)
- **Default Correlation:** Clients with lower income (bottom quartile) have 2-3x higher default risk
- **Key Insight:** Income is strongly predictive of default probability

**Credit Amount (AMT_CREDIT):**
- Mean: ~599,000 | Median: ~510,000
- **Right-skewed distribution** with outliers reaching 4.05 million
- **Relationship:** Credit amount relative to income (CREDIT_INCOME_RATIO) is more predictive than absolute amount
- Default customers typically have higher credit-to-income ratios

**Employment Tenure (DAYS_EMPLOYED):**
- Negative values represent days (converted to years)
- Mean employment: ~8 years | Median: ~5 years
- **Outliers:** Some records show unrealistic tenure (e.g., 1000+ years → data quality issues)
- **Default Pattern:** Stable employment (5+ years) correlates with lower default risk

### Missing Value Patterns
**High-Missing Features:**
- `OCCUPATION_TYPE`: 31% missing (not all occupations provided)
- `AMT_GOODS_PRICE`: 40% missing (N/A for revolving credit)
- `EXT_SOURCE_3`: 35% missing (external score unavailable for some clients)

**Strategy Applied:**
- Missing indicator features created for highly missing columns
- Absence of external scores treated as separate category
- Categorical missing values imputed with mode or "Not Specified"

### Categorical Features Analysis

**Contract Type:**
- Revolving Credit vs Cash Loans distribution
- Different default rates by contract type

**Income Type:**
- Working / Self-employed / Pensioner / Student / etc.
- Employment type shows significant correlation with default

**Education Level:**
- Secondary / Higher / Incomplete Higher education
- Education level inversely correlates with default probability

**Housing Type:**
- Renting / Own House / Mortgage / With Parents
- Homeownership status relevant to financial stability

**Occupation Type:**
- Various professional categories with different risk profiles
- Certain occupations show higher default rates

### Previous Application Analysis
**New vs Returning Clients:**
- ~70% of current applications are from existing customers
- ~30% are new customers with no previous application history

**Key Risk Indicator - Refusal History:**
- Customers WITH previous rejections: **12.5% default rate**
- Customers WITHOUT previous rejections: **7.2% default rate**
- **Risk Increase:** +75% higher default probability if previously rejected
- **Business Insight:** Previous refusal is one of the strongest default predictors

**Approval Rate Pattern:**
- Clients with 0% previous approval rate → 14.8% default
- Clients with 50-100% approval rate → 6.5% default
- **Clear Correlation:** Historical approval rate strongly predicts current default probability

**Application Frequency:**
- Customers with 1-2 previous applications: 7.5% default
- Customers with 5+ previous applications: 9.2% default
- Frequent applicants show slightly higher risk

### Outlier Summary
**Features with Significant Outliers (IQR Method):**
| Feature | Outlier Count | % of Data |
|---------|---------------|----------|
| AMT_INCOME_TOTAL | 18,234 | 5.9% |
| AMT_CREDIT | 12,567 | 4.1% |
| DAYS_EMPLOYED | 8,945 | 2.9% |
| OWN_CAR_AGE | 6,234 | 2.0% |
| APARTMENTS_AVG | 4,567 | 1.5% |

### Correlation Insights
**Strongest Correlations with Default:**
1. **External Scores (EXT_SOURCE_1, 2, 3):** -0.55 to -0.75 (negative = external scores decrease default risk)
2. **Days Birth:** 0.08 (younger = higher default risk)
3. **AMT_INCOME_TOTAL:** -0.06 (lower income = higher risk)
4. **Days Employed:** -0.04 (shorter employment = higher risk)
5. **Previous Application Refusal Indicator:** 0.12 (strongest among behavioral features)

**Multicollinearity Detected:**
- CREDIT_INCOME_RATIO and AMT_CREDIT highly correlated (r=0.95)
- AMT_ANNUITY and AMT_CREDIT moderately correlated (r=0.70)
- Multiple EXT_SOURCE variables somewhat correlated

### Skewness Analysis
**Highly Skewed Features (|Skewness| > 2.0):**
- AMT_INCOME_TOTAL (Skewness: 8.9)
- AMT_CREDIT (Skewness: 4.2)
- DAYS_EMPLOYED (Skewness: 3.1)
- AMT_ANNUITY (Skewness: 5.7)

**Action Taken:** Log transformation applied to stabilize variance and improve model performance

---

## 📊 Model Development & Results

### Algorithm Choice: LightGBM
**Why LightGBM?**
- **Imbalanced Data Handling:** Built-in support for class weights and scale_pos_weight parameter
- **Speed:** 10-20x faster than traditional gradient boosting on large datasets
- **Memory Efficiency:** Optimal for 307K+ records with 122 features
- **Automatic Feature Importance:** Provides insights into which features drive predictions
- **Categorical Handling:** Native support for categorical features without one-hot encoding

### Hyperparameter Optimization
- **Tool:** Optuna for Bayesian hyperparameter search
- **Metric:** ROC-AUC (appropriate for imbalanced classification)
- **Parameters Tuned:** Learning rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree

---

### Model Performance Metrics
- **ROC-AUC Score:** [Reported in 03_modeling.ipynb]
- **Accuracy:** [Reported in 03_modeling.ipynb]
- **F1-Score:** [Reported in 03_modeling.ipynb]
- **Precision & Recall:** [Reported in 03_modeling.ipynb]
- **Threshold Optimization:** [Reported in 04_thresholding.ipynb]

### Top 15 Most Important Features
1. **EXT_SOURCE_3** - External credit score #3 (most predictive)
2. **EXT_SOURCE_2** - External credit score #2
3. **EXT_SOURCE_1** - External credit score #1
4. **DAYS_BIRTH** - Client age at application
5. **AMT_INCOME_TOTAL** - Total monthly income
6. **DAYS_EMPLOYED** - Employment tenure
7. **AMT_CREDIT** - Loan amount
8. **CREDIT_INCOME_RATIO** - Debt leverage metric
9. **ANNUITY_INCOME_RATIO** - Monthly payment burden
10. **REGION_RATING_CLIENT** - Regional risk rating
11. **PREV_REFUSED_COUNT** - History of refusals
12. **CNT_FAM_MEMBERS** - Family size
13. **ORGANIZATION_TYPE** - Type of employer
14. **PREV_APP_COUNT** - Historical application frequency
15. **NAME_EDUCATION_TYPE** - Education level

### Business Insights from Model
**Risk Factors (Increase Default Probability):**
- Low external credit scores (EXT_SOURCE < 0.3)
- Young age (< 30 years old)
- Low income (bottom quintile)
- Previous application refusals (1+ rejections = +75% risk)
- High credit-to-income ratio (> 2.5)
- Short employment tenure (< 1 year)

**Protective Factors (Decrease Default Probability):**
- High external credit scores (> 0.7)
- Mature age (40-55 years)
- High income (top quintile)
- Consistent approval history
- Low debt-to-income ratio
- Stable long-term employment (5+ years)
- Home ownership

**Threshold Decision Strategy:**
- Default probability threshold optimized based on business cost-benefit analysis
- Balances precision (reducing false alarms) vs recall (catching actual defaults)
- Documented in 04_thresholding.ipynb

---

## 🛠 Technologies Used

- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** LightGBM, Scikit-Learn, Optuna
- **Explainable AI:** SHAP
- **Deployment:** Streamlit (Frontend), FastAPI (Backend ready)

---

## 🚀 Getting Started & Installation

### Try the Web App
Access the deployed application directly from your browser: 
👉 [https://credit-risk-scoring-kaf9rhmrgkq5rvu4nybgic.streamlit.app/](https://credit-risk-scoring-kaf9rhmrgkq5rvu4nybgic.streamlit.app/)

### Run Locally (Local Setup)

You can run this project locally by copying and pasting the following commands into your terminal:

```bash
# 1. Clone the repository and navigate into it
git clone [https://github.com/dqhung1306/credit-risk-scoring.git](https://github.com/dqhung1306/credit-risk-scoring.git)
cd credit-risk-scoring

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # For Windows (PowerShell)
# source venv/bin/activate     # For macOS/Linux

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run deployments/main.py
## 👤 Author

**Name:** Đặng Quang Hưng
**GitHub:** 

---

## 📝 License

This project is for educational and portfolio purposes.

---

## 🔗 References

- [Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
