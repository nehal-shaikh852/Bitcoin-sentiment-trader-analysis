# Bitcoin Market Sentiment vs Trader Performance Analysis

## Understanding the Relationship Between Market Sentiment and Trading Behavior

---

# Project Overview

This project explores the relationship between Bitcoin market sentiment and trader performance using two primary datasets:

1. **Bitcoin Market Sentiment Dataset**

   - Contains Fear & Greed sentiment classifications.
   - Key columns include:
     - `date`
     - `classification`
     - `value`
     - `timestamp`

2. **Historical Trader Data from Hyperliquid**

   - Contains detailed cryptocurrency trading records.
   - Key columns include:
     - `Account`
     - `Coin`
     - `Execution Price`
     - `Size Tokens`
     - `Size USD`
     - `Side`
     - `Timestamp`
     - `Start Position`
     - `Direction`
     - `Closed PnL`
     - `Fee`
     - `Trade ID`
     - `Leverage`
     - and additional trading metrics.

The primary goal of this project is to uncover hidden patterns between market sentiment and trader behavior while generating actionable insights that can support smarter trading strategies and data-driven decision-making.

---

# Project Objectives

The main objectives of this analysis are:

- Perform Exploratory Data Analysis (EDA)
- Analyze the distribution of Bitcoin market sentiment
- Study trader profitability across different market conditions
- Merge sentiment data with trading activity
- Identify behavioral trading patterns
- Build a machine learning model to predict profitable trades
- Generate actionable insights from the analysis

---

# Technologies and Tools Used

The following technologies and libraries were used throughout the project:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Project Workflow

## 1. Data Loading

Both datasets were loaded using the Pandas library.

```python
trades_data = pd.read_csv("historical_data.csv")
sentiment = pd.read_csv("fear_greed_index.csv")
```

This step ensured that the datasets were available for preprocessing and analysis.

---

## 2. Exploratory Data Analysis (EDA)

### Sentiment Dataset Analysis

The following checks and analyses were performed:

- Dataset inspection
- Missing value detection
- Shape and datatype verification
- Sentiment distribution analysis

Example:

```python
print(sentiment.info())
print(sentiment.shape)
print(sentiment['classification'].value_counts())
```

### Key Findings

The market sentiment dataset contained the following sentiment categories:

- Fear
- Greed
- Extreme Fear
- Extreme Greed
- Neutral

Additional observations:

- Sentiment values fluctuated significantly over time.
- The market frequently transitioned between fear and greed phases.
- Extreme sentiment periods often aligned with strong market volatility.

---

## 3. Sentiment Visualization

### Sentiment Trend Over Time

A time-series graph was created to visualize changes in market sentiment.

```python
plt.figure(figsize=(12,6))
plt.plot(sentiment['date'], sentiment['value'])
plt.title("Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Value")
plt.xticks(rotation=45)
plt.show()
```

### Insights from Visualization

- High volatility was observed in sentiment values.
- Repeated transitions between fear and greed phases were visible.
- Extreme fear and extreme greed periods coincided with major market movements.

---

## 4. Trader Dataset Analysis

A detailed inspection of trader execution data was performed.

```python
print(trades_data.info())
print(trades_data.shape)
print(trades_data['Closed PnL'].describe())
```

### Key Observations

- The dataset contained a large number of trading records.
- Trader profitability varied significantly.
- Both profitable and loss-making trades were present.
- Trade sizes and leverage values showed a wide distribution.

---

## 5. Data Cleaning and Preprocessing

Several preprocessing steps were performed to prepare the datasets for analysis:

- Date conversion
- Timestamp formatting
- Removing unnecessary columns
- Handling missing values
- Preparing features for modeling

Example:

```python
trades_data['date'] = pd.to_datetime(trades_data['Timestamp IST']).dt.date
sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
```

Proper preprocessing improved data consistency and ensured accurate analysis.

---

## 6. Dataset Merging

The trader dataset was merged with the sentiment dataset using the `date` column.

```python
merged_df = pd.merge(
    cleaned_trades_data,
    sentiment,
    on='date',
    how='inner'
)
```

This step enabled direct comparison between trading activity and market sentiment.

---

# Sentiment vs Trading Analysis

## Average Profitability by Sentiment

The average profitability for each sentiment category was analyzed.

```python
merged_df.groupby('classification')['Closed PnL'].mean()
```

### Findings

Traders generally performed better during:

- Extreme Greed
- Greed

Lower profitability was observed during:

- Fear
- Neutral market conditions

These findings indicate that market sentiment strongly influences trader outcomes and risk appetite.

---

## Average Trade Size by Sentiment

Trade size behavior was analyzed across sentiment categories.

```python
merged_df.groupby('classification')['Size USD'].mean()
```

### Findings

- Trade sizes increased during greed phases.
- Traders became more aggressive in bullish market conditions.
- Smaller and more cautious positions were observed during fearful markets.

---

# Correlation Analysis

A correlation heatmap was generated to identify relationships between numerical variables.

```python
sns.heatmap(
    merged_df.corr(numeric_only=True),
    annot=True
)
```

### Insights

- Trade size and profitability showed meaningful relationships.
- Sentiment values influenced trading behavior.
- Leverage and risk exposure varied under different market conditions.

---

# Machine Learning Model

## Objective

The objective of the machine learning model was to predict whether a trade would be profitable based on sentiment and trading-related features.

---

## Target Variable Creation

A binary target variable was created:

```python
merged_df['profitable'] = (
    merged_df['Closed PnL'] > 0
).astype(int)
```

- `1` indicates a profitable trade.
- `0` indicates a non-profitable trade.

---

## Feature Selection

Relevant features were selected for training the model.

```python
X = merged_df.drop(
    columns=[
        'Closed PnL',
        'profitable',
        'Timestamp',
        'datetime',
        'date'
    ],
    errors='ignore'
)
```

Target variable:

```python
y = merged_df['profitable']
```

---

## Model Selection

A Random Forest Classifier was used for classification.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The Random Forest algorithm was selected because of its strong performance on structured datasets and ability to handle complex feature relationships.

---

## Train-Test Split

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Model Evaluation

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
```

### Model Performance

Approximate model performance results:

- Accuracy: \~96%
- Precision: \~94%
- Recall: \~97%
- F1 Score: \~96%

The model achieved strong predictive performance in identifying profitable trades.

---

# Final Insights and Business Understanding

## 1. Market Sentiment Influences Trader Performance

- Higher profitability was observed during greed phases.
- Lower profitability occurred during fear-driven markets.

## 2. Trader Behavior Changes with Sentiment

- Traders took larger positions during bullish conditions.
- More cautious behavior was observed during fearful sentiment.

## 3. Sentiment Works as a Predictive Signal

- Market sentiment improved trading analysis.
- Combining sentiment with trading metrics enhanced predictive capability.

## 4. Machine Learning Improved Trade Prediction

- The Random Forest model delivered strong classification results.
- Predictive modeling can support better strategy development and risk management.



---

# Conclusion

This project successfully explored the relationship between Bitcoin market sentiment and trader performance.

By combining sentiment analysis with historical trading data, the analysis uncovered valuable behavioral patterns and demonstrated how sentiment indicators can improve predictive modeling.

Key achievements of the project include:

- Identification of sentiment-driven trading behavior
- Analysis of trader risk appetite
- Strong machine learning model performance
- Generation of actionable insights for smarter trading strategies

Overall, the project demonstrates how sentiment analysis can significantly enhance trading intelligence, profitability analysis, and decision-making in cryptocurrency markets.

