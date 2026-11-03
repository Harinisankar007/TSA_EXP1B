# Ex.No: 1B CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.

### DATASET
DailyDelhiClimate

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
train = pd.read_csv("DailyDelhiClimateTrain.csv")
test = pd.read_csv("DailyDelhiClimateTest.csv")
```
```python
for df in (train, test):
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime( df["date"], errors="coerce")
    df["meantemp"] = pd.to_numeric(df["meantemp"], errors="coerce")
    df["meantemp"] = df["meantemp"].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=["date"], inplace=True)

# Combine files for descriptive analysis, not forecasting
data = pd.concat([train[["date", "meantemp"]],test[["date", "meantemp"]]],ignore_index=True)

# If a date appears in both files, keep the test-file row
data.drop_duplicates(
    subset="date", keep="last", inplace=True
)

data.sort_values("date", inplace=True)
data.set_index("date", inplace=True)

# Ensure a regular daily index
data = data.asfreq("D")

# Fill missing temperatures for decomposition
data["meantemp"] = data["meantemp"].interpolate(method="time",limit_direction="both")

if data["meantemp"].isna().any():
    raise ValueError("No valid mean-temperature values were found.")
# Log transformation requires positive values
if (data["meantemp"] <= 0).any():
    raise ValueError(
        "Log transformation requires positive meantemp values. "
        "Your dataset contains zero or negative temperatures."
    )

period = 365

# One extra observation is needed because differencing
# removes the first observation.
if len(data) - 1 < 2 * period:
    raise ValueError(
        "At least 731 daily observations are required "
        "for this code with period=365."
    )

print("Original Data:")
print(data.head())
```
<img width="625" height="165" alt="image" src="https://github.com/user-attachments/assets/e091cdde-7503-428e-bdd3-c57c7389e78c" />

```python
data["temp_diff"] = (data["meantemp"] - data["meantemp"].shift(1))
result = seasonal_decompose(data["meantemp"],model="additive", period=period)
data["temp_residual"] = result.resid
data["temp_log"] = np.log(data["meantemp"])
data["temp_log_diff"] = ( data["temp_log"] - data["temp_log"].shift(1))
```
```
log_diff_result = seasonal_decompose(data["temp_log_diff"].dropna(), model="additive", period=period)
data["temp_log_diff_residual"] = log_diff_result.resid
print("\nTransformed Data:")
print(data.head(10).round(4).to_string())
```
<img width="806" height="259" alt="image" src="https://github.com/user-attachments/assets/5c0dbb3c-d6c5-45f9-95a4-83ddbe26dfe3" /><BR>
```python



plots = [
    (
        "meantemp",
        "Original Data",
        "Original Mean Temperature",
        "blue"
    ),
    (
        "temp_diff",
        "Regular Differencing",
        "Regular Difference",
        "purple"
    ),
    (
        "temp_residual",
        "Residuals After Removing Trend and Seasonality",
        "Decomposition Residual",
        "green"
    ),
    (
        "temp_log",
        "Log Transformation",
        "Log Transformation",
        "orange"
    ),
    (
        "temp_log_diff",
        "Log Transformation and Regular Differencing",
        "Log Difference",
        "red"
    ),
    (
        "temp_log_diff_residual",
        "Decomposition Residuals of Log-Differenced Data",
        "Log-Difference Residual",
        "brown"
    )
]

fig, axes = plt.subplots( 6, 1,figsize=(16, 24))

for ax, (column, title, label, color) in zip(axes, plots):
    ax.plot(
        data.index,
        data[column],
        color=color,
        linewidth=1,
        label=label
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Temp")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
```

### OUTPUT:

## ORIGINAL DATA

<img width="1174" height="291" alt="image" src="https://github.com/user-attachments/assets/d43b1054-b142-466d-b616-de604a7c74cf" />


## REGULAR DIFFERENCING:

<img width="1180" height="298" alt="image" src="https://github.com/user-attachments/assets/f13a3ec5-e138-4dc2-a670-b2248bbdd887" />


## SEASONAL ADJUSTMENT:
<img width="1154" height="306" alt="image" src="https://github.com/user-attachments/assets/7dd66f70-7ccf-4ab1-ad7d-3777459a4410" />


## LOG TRANSFORMATION:

<img width="1172" height="319" alt="image" src="https://github.com/user-attachments/assets/d7e24e28-9e87-46c8-8935-a018a8e966b4" />

## FINAL COMPARISON

<img width="1205" height="448" alt="image" src="https://github.com/user-attachments/assets/a60f3c65-939d-4285-bd01-ca105f2c3629" />


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
