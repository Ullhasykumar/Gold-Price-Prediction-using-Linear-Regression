# Gold Price Prediction using Linear Regression (From Scratch)

A machine learning project that predicts the **closing price of Gold (GLD ETF)** using a custom Linear Regression model built entirely from scratch with NumPy — no sklearn.LinearRegression used.

---

## Demo / Output

| Metric | Value |
|--------|-------|
| R² Score | ~0.99+ |
| Mean Squared Error | Very Low (< 5) |

> The model learns to predict Gold's closing price from the same day's High, Low, and Open prices — achieving near-perfect accuracy due to the strong intra-day correlation in financial data.

---

##  Features

- **Custom Linear Regression** — implemented using the **Normal Equation** (closed-form solution), not gradient descent
-  **Real-time financial data** — fetched directly via `yfinance` (no manual CSV needed)
-  **Multi-feature regression** — uses `High`, `Low`, and `Open` as input features
-  **Model evaluation** — uses R² Score and Mean Squared Error
-  **Comparison table** — side-by-side view of actual vs. predicted closing prices
-  **No ML black box** — every step of the math is transparent and readable

---

## How It Works

### The Math Behind It

This project solves Linear Regression using the **Normal Equation**:

```
β = (XᵀX)⁻¹ · Xᵀ · y
```

Where:
- `X` = feature matrix (High, Low, Open) with a bias column of 1s prepended
- `y` = target vector (Close prices)
- `β` = learned coefficients (intercept + weights for each feature)

This gives the **exact optimal solution** in one step — no iteration, no learning rate needed.

### Prediction

Once `β` is learned:
```
ŷ = X · coef_ + intercept_
```

---

##  Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gold-price-prediction.git
cd gold-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python gold_price_prediction.py
```

> The script will automatically download GLD data from Yahoo Finance for the date range `2020-01-01` to `2026-06-30`.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations & Normal Equation |
| `pandas` | DataFrame manipulation |
| `matplotlib` | (Available for future plotting) |
| `yfinance` | Fetching live GLD ETF data from Yahoo Finance |
| `scikit-learn` | `train_test_split`, `r2_score`, `mean_squared_error` |

---

## Code Walkthrough

### Step 1 — Data Collection
```python
gold_data = yf.download("GLD", start='2020-01-01', end='2026-06-30')
```
Downloads daily OHLCV data for the **SPDR Gold Shares ETF (GLD)** directly from Yahoo Finance.

### Step 2 — Feature Selection
```python
X, y = gold[['High', 'Low', 'Open']], gold['Close']
```
Three features are selected: `High`, `Low`, `Open`. The target is `Close`.

### Step 3 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
80% of data for training, 20% for testing. `random_state=2` ensures reproducibility.

### Step 4 — Custom Linear Regression (Normal Equation)
```python
class myLR:
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)           # Add bias column
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
```
Solves for the optimal coefficients in closed form — mathematically equivalent to sklearn's `LinearRegression` but written from scratch.

### Step 5 — Evaluation
```python
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
```

---

## Scope of Improvements

This project is a clean baseline. Here are meaningful ways to extend it:

###  Model Improvements
- [ ] **Add more features** — Volume, Moving Averages (MA5, MA20), RSI, MACD, Bollinger Bands
- [ ] **Try other models** — Random Forest, XGBoost, SVR, LSTM (for time-series awareness)
- [ ] **Gradient Descent implementation** — Rewrite `fit()` using iterative optimization instead of the Normal Equation to handle larger datasets more efficiently
- [ ] **Regularization** — Add L1 (Lasso) or L2 (Ridge) penalty to avoid overfitting on noisy financial data

###  Data / Time-Series Improvements
- [ ] **Avoid data leakage** — The current train/test split is random, which is problematic for time-series. Use **chronological split** (e.g., train on 2020–2024, test on 2025–2026)
- [ ] **Lag features** — Add previous day's Close, High, Low as features to capture temporal patterns
- [ ] **External signals** — Incorporate USD Index (DXY), S&P 500, inflation data, or interest rates

---

## Why Does R² ≈ 0.99?

The very high R² is expected — not a bug. Within a single trading day, a stock/ETF's `High`, `Low`, and `Open` prices are **extremely strongly correlated** with its `Close` price. This is a valid regression task, but the features are intra-day (not predictive of future days). For **true forecasting**, you'd need features from *previous* days to predict *today's* close.

