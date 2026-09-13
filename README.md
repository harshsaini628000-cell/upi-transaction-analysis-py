

```markdown
# UPI Transactions Exploratory Data Analysis & Modeling (Google Colab)

An end-to-end data analysis pipeline built using **Python**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn** in **Google Colab** to process, clean, transform, analyze, and visualize 4,000 simulated UPI transaction records.

---

## 1. Project Overview & Dataset Attributes

The dataset `upi_transactions_4000.csv` contains 4,000 transaction records across 14 initial features:

- **Transaction_ID**: Unique alphanumeric transaction identifier (e.g., `UPI100000001`).
- **Date**: Date string in `YYYY-MM-DD` format.
- **Time**: Time string in `HH:MM:SS` format.
- **Sender_Bank**: Remitting financial institution (e.g., HDFC, ICICI, SBI).
- **Receiver_Bank**: Beneficiary financial institution.
- **UPI_App**: Payment application (Google Pay, PhonePe, Paytm, CRED, etc.).
- **Transaction_Type**: Nature of transaction (`P2P`, `P2M`, `Recharge`, `Bill Payment`).
- **Amount_INR**: Monetary value in Indian Rupees (INR).
- **Status**: Final transaction state (`Success`, `Failed`, `Reversed`).
- **Failure_Reason**: Reason code logged when a transaction fails.
- **City**: Geographic origin of the request.
- **Device_Type**: Client hardware platform (`Android`, `iOS`).
- **Response_Time_ms**: End-to-end network latency in milliseconds.
- **Is_Weekend**: Boolean flag indicating if the transaction occurred on a weekend.

---

## 2. Running on Google Colab

Follow these quick steps to run the project in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Click the **Folder icon** 📁 on the left toolbar.
3. Click the **Upload to session storage** button and select `upi_transactions_4000.csv` (or use the upload code snippet in Step 1).
4. Run the code cells sequentially by pressing `Shift + Enter`.

---

## 3. Line-by-Line Code Breakdown

### Step 1: File Ingestion in Google Colab

```python
from google.colab import files
uploaded = files.upload()

```

* `from google.colab import files`: Imports the official Google Colab file management library.
* `uploaded = files.upload()`: Spawns an interactive file upload button in the Colab cell output, allowing you to select and upload `upi_transactions_4000.csv` from your computer into Colab's cloud environment.

```python
import pandas as pd
df = pd.read_csv("upi_transactions_4000.csv")
df.head(5)

```

* `import pandas as pd`: Loads the Pandas data manipulation library.
* `df = pd.read_csv("upi_transactions_4000.csv")`: Reads the uploaded CSV file from Colab's current session directory into a 2D Pandas DataFrame named `df`.
* `df.head(5)`: Displays the first 5 records (rows 0 to 4) as an interactive Colab table.

---

### Step 2: Datetime Feature Engineering

```python
df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df['Hour'] = df['Timestamp'].dt.hour
df['Day_Name'] = df['Timestamp'].dt.day_name()
df['Month'] = df['Timestamp'].dt.month
df.head()

```

* `df['Timestamp'] = pd.to_datetime(...)`: Joins the separate `Date` and `Time` string columns with a space and converts them into standardized `datetime64` timestamps.
* `df['Hour'] = df['Timestamp'].dt.hour`: Uses the `.dt` accessor to extract the hour (0 to 23).
* `df['Day_Name'] = df['Timestamp'].dt.day_name()`: Extracts the full day name (e.g., Sunday, Monday).
* `df['Month'] = df['Timestamp'].dt.month`: Extracts the month integer (1 to 12).
* `df.head()`: Previews the updated DataFrame with the new temporal features.

---

### Step 3: Missing Value Verification & Imputation

```python
failed_count = (df['Status'] == 'falied').sum()
failure_reason_count = df['Failure_Reason'].notna().sum()
print(f"Number of failed transactions: {failed_count}")
print(f"Number of null values in Failure_Reason: {failure_reason_count}")

```

* `failed_count = (df['Status'] == 'falied').sum()`: Checks rows matching status (returns 0 due to the typo `'falied'` instead of `'Failed'`).
* `failure_reason_count = df['Failure_Reason'].notna().sum()`: Counts rows where `Failure_Reason` is recorded (1,667 rows).
* `print(...)`: Prints diagnostic statistics in the Colab output cell.

```python
df.loc[
    (df['Status'] == 'Success') &
    (df['Failure_Reason'].isna()),
    'Failure_Reason'
] = 'Not Applicable'

```

* `df.loc[...] = 'Not Applicable'`: Applies conditional label-based indexing:
* `df['Status'] == 'Success'`: Filters successful transactions.
* `df['Failure_Reason'].isna()`: Identifies empty `NaN` reason cells.
* Sets the `Failure_Reason` for all successful transactions to `'Not Applicable'`.



```python
df['Failure_Reason'].isna().sum()

```

* `df['Failure_Reason'].isna().sum()`: Verifies remaining missing values (303 unrecorded failure reasons for failed/reversed transactions).

---

### Step 4: Slicing High-Value Failed Transactions

```python
subset = df[(df['Amount_INR'] > 5000) & (df['Status'] == 'Failed')]
subset.head()

```

* `subset = df[...]`: Filters transactions where `Amount_INR > 5000` AND `Status == 'Failed'`.
* `subset.head()`: Displays the first 5 records of high-value failed payments.

---

### Step 5: Binary Encoding for Success Rate

```python
df['Is_Success'] = (df['Status'] == 'Success').astype(int)
df.head()

```

* `(df['Status'] == 'Success')`: Converts the status check to boolean `True` or `False`.
* `.astype(int)`: Encodes `True` as `1` and `False` as `0`.
* `df['Is_Success'] = ...`: Stores the result in a binary column, allowing `.mean()` to compute the overall success rate.

---

### Step 6: Log Transformation for Skewed Amounts

```python
import numpy as np
df['Log_Amount_INR'] = np.log1p(df['Amount_INR'])
df.head(2)

```

* `import numpy as np`: Imports NumPy for array and mathematical operations.
* `np.log1p(df['Amount_INR'])`: Computes $\ln(1 + x)$ to normalize the right-skewed payment distribution and avoid division-by-zero errors.
* `df.head(2)`: Checks the first 2 rows of the new `Log_Amount_INR` column.

---

### Step 7: Latency Outlier Detection via IQR

```python
df['Response_Time_ms'].describe()

```

* `df['Response_Time_ms'].describe()`: Outputs 8-point summary statistics: mean (844.25 ms), std dev (774.28 ms), min (50 ms), 25% (285 ms), median (614 ms), 75% (1124 ms), and max (6070 ms).

```python
Q1 = np.percentile(df['Response_Time_ms'], 25)
Q3 = np.percentile(df['Response_Time_ms'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['Response_Time_ms'] < lower_bound) |
    (df['Response_Time_ms'] > upper_bound)
]

```

* `Q1`: 25th percentile (285.0 ms).
* `Q3`: 75th percentile (1124.0 ms).
* `IQR`: Interquartile Range ($Q3 - Q1 = 839.0\text{ ms}$).
* `lower_bound` / `upper_bound`: Calculates Tukey bounds: Lower = -973.5 ms, Upper = 2382.5 ms.
* `outliers`: Filters transactions with response times $> 2382.5\text{ ms}$ (218 outlier events).

---

### Step 8: Conditional Binning (Performance Tiers)

```python
conditions = [
    df['Response_Time_ms'] < 300,
    (df['Response_Time_ms'] >= 300) & (df['Response_Time_ms'] <= 700),
    df['Response_Time_ms'] > 700
]
choices = ['Fast', 'Moderate', 'Slow']
df['Performance_Tier'] = np.select(conditions, choices, default='')
df[['Response_Time_ms', 'Performance_Tier']].head()

```

* `conditions`: Defines three performance ranges (< 300 ms, 300–700 ms, > 700 ms).
* `choices`: Assigns labels (`'Fast'`, `'Moderate'`, `'Slow'`).
* `np.select(...)`: Applies vectorized conditional logic to populate the `Performance_Tier` column.

---

### Step 9: Correlation & Covariance Matrices

```python
corr_matrix = np.corrcoef(df['Amount_INR'], df['Response_Time_ms'])
cov_matrix = np.cov(df['Amount_INR'], df['Response_Time_ms'])

```

* `np.corrcoef(...)`: Computes Pearson's correlation coefficient matrix ($r \approx 0.00196$, showing zero linear relationship between amount and latency).
* `np.cov(...)`: Computes sample covariance between transaction value and network latency.

---

### Step 10: Inter-Bank Success Rate Pivot Table

```python
bank_success_matrix = pd.pivot_table(
    df,
    values='Is_Success',
    index='Sender_Bank',
    columns='Receiver_Bank',
    aggfunc='mean'
)

```

* `pd.pivot_table(...)`: Builds a 12x12 inter-bank grid:
* `Sender_Bank` as row index.
* `Receiver_Bank` as column headers.
* `aggfunc='mean'`: Calculates the average clearance success rate per bank pair route.



---

### Step 11: UPI App Market Share Aggregation

```python
group_by = df.groupby('UPI_App')
total_volume = group_by['Transaction_ID'].count()
total_value = group_by['Amount_INR'].sum()
average_failure_rate = group_by['Is_Success'].mean

```

* `group_by = df.groupby('UPI_App')`: Groups transactions by the 8 payment apps.
* `total_volume`: Total transactions per app.
* `total_value`: Total processed amount in INR per app.
* `average_failure_rate`: Points to the mean calculation method (use `.mean()` with parentheses to return the calculated Series).

---

### Step 12: Peak Hour Analysis

```python
peak_hour = df.groupby('Hour').agg(
    Transaction_Count=('Hour', 'count'),
    Avg_Response_Time_ms=('Response_Time_ms', 'mean')
).reset_index()

```

* `.groupby('Hour')`: Groups records across all 24 hours of the day (0 to 23).
* `.agg(...)`: Computes:
* `Transaction_Count`: Total transactions in that hour.
* `Avg_Response_Time_ms`: Average latency in that hour.


* `.reset_index()`: Resets `Hour` to a standard DataFrame column.

---

### Step 13: 24-Hour Trend Line Plot (Matplotlib)

```python
import matplotlib.pyplot as plt
import seaborn as sns

max_transactions = peak_hour['Transaction_Count'].max()
peak_hours = peak_hour[peak_hour['Transaction_Count'] == max_transactions]

plt.figure(figsize=(14, 6))
plt.plot(peak_hour['Hour'], peak_hour['Transaction_Count'], marker='o', linewidth=2, label='Transaction Volume')
plt.scatter(peak_hours['Hour'], peak_hours['Transaction_Count'], s=120, zorder=5, label='Peak Hour')

for _, row in peak_hours.iterrows():
    plt.annotate(
        f"Peak: {int(row['Hour']):02d}:00\n{int(row['Transaction_Count'])} transactions",
        (row['Hour'], row['Transaction_Count']),
        xytext=(0, 15),
        textcoords='offset points',
        ha='center'
    )

plt.xticks(range(24))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Hour of Day')
plt.ylabel('Transaction Count')
plt.title('Hourly Transaction Volume (24-Hour Cycle)')
plt.legend()
plt.tight_layout()
plt.show()

```

* `plt.figure(figsize=(14, 6))`: Initializes a 14x6 inch figure canvas inside Colab.
* `plt.plot(...)`: Draws the transaction volume curve.
* `plt.scatter(...)`: Highlights the peak hour (Hour 16 with 194 transactions).
* `plt.annotate(...)`: Adds a callout bubble for the peak.
* `plt.xticks(range(24))`: Shows all 24 hour markers along the x-axis.
* `plt.show()`: Renders the chart inline in the Colab cell.

---

### Step 14: Dual-Pane Distribution Histograms

```python
log_amount = np.log1p(df['Amount_INR'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Amount_INR'], bins=30)
axes[0].set_title('Distribution of Amount_INR')
axes[0].set_xlabel('Amount (INR)')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(log_amount, bins=30)
mean_log = log_amount.mean()
median_log = log_amount.median()
axes[1].axvline(mean_log, linestyle='--', label=f'Mean = {mean_log:.2f}')
axes[1].axvline(median_log, linestyle='-', label=f'Median = {median_log:.2f}')
axes[1].set_title('Log-Transformed Amount_INR')
axes[1].set_xlabel('log1p(Amount_INR)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

```

* `plt.subplots(1, 2, figsize=(14, 5))`: Creates side-by-side subplots.
* `axes[0].hist(...)`: Visualizes raw right-skewed payment amounts.
* `axes[1].hist(...)`: Shows normalized log-transformed amounts.
* `axes[1].axvline(...)`: Adds vertical reference lines for mean and median.

---

### Step 15: Stacked Bar Chart by Device Type

```python
status_counts = pd.crosstab(df['Device_Type'], df['Status'])
status_percent = status_counts.div(status_counts.sum(axis=1), axis=0) * 100

status_percent[['Success', 'Failed', 'Reversed']].plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6)
)
plt.title('Transaction Status Proportion by Device Type')
plt.xlabel('Device Type')
plt.ylabel('Proportion (%)')
plt.xticks(rotation=0)
plt.legend(title='Transaction Status')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

```

* `pd.crosstab(...)`: Creates a contingency table of device type vs. transaction status.
* `.div(..., axis=0) * 100`: Converts counts into row-wise percentages summing to 100%.
* `.plot(kind='bar', stacked=True)`: Renders a 100% stacked bar chart comparing Android and iOS.

---

### Step 16: Latency Distribution Boxplot (Seaborn)

```python
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x='UPI_App',
    y='Response_Time_ms'
)
plt.title('Response Time Distribution by UPI App')
plt.xlabel('UPI App')
plt.ylabel('Response Time (ms)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

```

* `sns.boxplot(...)`: Generates boxplots showing median, interquartile ranges, and outliers for each UPI app.
* `plt.xticks(rotation=45)`: Tilts app labels 45 degrees for clean readability.

```

```
