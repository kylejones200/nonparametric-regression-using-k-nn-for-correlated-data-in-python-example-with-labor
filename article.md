---
author: "Kyle Jones"
date_published: "March 30, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/nonparametric-regression-using-k-nn-for-correlated-data-in-python-example-with-labor-1cb71a84479b"
---

# Nonparametric Regression using k-NN for Correlated Data in Python

(example with labor... In practice, data rarely arrive as clean, independent observations. Instead, we deal with sequences --- sensor logs, financial returns, or...

### Nonparametric Regression using k-NN for Correlated Data in Python (example with labor participation rate)
In practice, data rarely arrive as clean, independent observations. Instead, we deal with sequences --- sensor logs, financial returns, or monthly labor force reports --- where one point affects the next. This chapter tackles nonparametric regression in such correlated settings. We'll focus on stationary sequences and uniform mixing, and show how familiar techniques like kernel smoothing and k-nearest neighbors (k-NN) extend to handle these challenges. We'll also dig into Python implementations with real-world time series.

### Regression for Dependent Sequences
Let (Xt,Yt) be a strictly stationary sequence: the joint distribution is invariant under time shifts. We assume dependence weakens over time but does not disappear. Our goal is to estimate:

m(x)=E\[Yt∣Xt=x\]

using methods that remain valid under correlation. The usual tricks for independent data break down unless we adjust for the dependence structure.

To control the dependency, we require a uniform mixing condition to ensure distant observations behave almost independently, so large-sample approximations (like the Central Limit Theorem) still apply.

k-NN estimates the regression function by averaging over the k closest neighbors.

In the dependent case, consistency still holds under mixing, but with slower convergence and inflated variance due to autocorrelation. We must ensure that k→∞ and k/n→0 while mixing coefficients decay fast enough --- ideally geometrically.

**Labor Force Participation (CIVPART)**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.nonparametric.kernel_regression import KernelReg

# Tufte-style plot setup
def tufte_style():
    plt.rcParams.update({
    'axes.grid': False,
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

tufte_style()

# Load data from FRED
series_id = 'CIVPART'
data = DataReader(series_id, 'fred', start='1948-01-01').dropna()
data.columns = ['Value']

# Kernel Regression Smoothing
data_reset = data.reset_index()
data_reset['Time'] = np.arange(len(data_reset))
kr = KernelReg(endog=data_reset['Value'], exog=data_reset['Time'], var_type='c', bw='cv_ls')
y_smooth, _ = kr.fit(data_reset['Time'])

# Plot Smoothed Series
plt.figure(figsize=(12, 6))
plt.plot(data_reset['DATE'], data_reset['Value'], label='Original', alpha=0.4, color='black')
plt.plot(data_reset['DATE'], y_smooth, label='Smoothed', color='red')
plt.xlabel('Date')
plt.ylabel('Participation Rate (%)')
plt.title('Smoothed Labor Force Participation Rate using Kernel Regression')
plt.legend()
plt.savefig('civpart_kernel_mape.png')
plt.show()

# Print MAPE
mape = mean_absolute_percentage_error(data_reset['Value'], y_smooth)
print(f'MAPE (Smoothed vs. Original): {mape:.4f}')
```


MAPE is 0.0012 which is what we would expect for a smoothing method Smoothing helps us understand the history but isn't useful for prediction.

The Nadaraya-Watson estimator smooths the data with dependent inputs.

For mixing sequences, this remains consistent. Bias stays at O(h²), but the variance reflects autocovariance across lags. Standard bandwidth selection methods may underperform, as they often assume independence.

**Example: Kernel Smoothing of Labor Force Data**

We'll predict the labor force participation rate using lagged values. First, we create lag features, then apply k-NN.

``` 
# Prepare lagged features for k-NN regression
for lag in range(1, 8):
    data[f'lag_{lag}'] = data['Value'].shift(lag)
data.dropna(inplace=True)

# Train-test split
split_date = '2015-01-01'
train = data[:split_date]
test = data[split_date:]
X_train = train.drop(['Value'], axis=1)
y_train = train['Value']
X_test = test.drop(['Value'], axis=1)
y_test = test['Value']

# Train k-NN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluate and plot k-NN predictions
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f'Mean Squared Error: {mse_knn:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='black', linestyle='-', alpha=0.6)
plt.plot(y_test.index, y_pred_knn, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Participation Rate (%)')
plt.title('k-NN Forecast: Civilian Labor Force Participation Rate')
plt.legend()
plt.savefig('civpart_knn_forecast.png')
plt.show()
```


This k-NN regression does a pretty good job of matching the data. It obviously misses the decrease in labor participation that was caused by COVID-19 in 2020.

### What If the Errors Are Dependent?
There are a few things we can do to deal with serially dependent errors. We can use Newey-West estimators to adjust for autocorrelation. We can apply block bootstrapping to respect the structure. We can use subsampling to thin out correlation but this causes loses information. We can choose the bandwidth parameter to correct for over-smoothing or under-smoothing in the presence of correlation.

Nonparametric regression methods can handle correlation. Mixing conditions like uniform mixing let us extend theoretical guarantees to k-NN and kernel smoothers. But correlation inflates variance, complicates inference, and warps bandwidth choice. When errors are correlated, adjustments like block bootstrap or long-run variance estimation are essential. In practice, these tools let us work with data as they are: messy, temporal, and richly dependent.
