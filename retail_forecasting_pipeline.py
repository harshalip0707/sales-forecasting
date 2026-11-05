"""
Retail Sales Forecasting & BI pipeline
--------------------------------------
Steps: load/clean -> EDA -> profitability -> seasonality -> forecasting -> Power BI exports

Install:  pip install pandas numpy matplotlib statsmodels scikit-learn xlrd openpyxl
Optional: pip install prophet xgboost lightgbm   (used automatically if present)

Run:      python retail_forecasting_pipeline.py
Default dataset: Kaggle "Sample - Superstore.csv". For other data, edit CONFIG only.
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
CONFIG = {
    "data_path": "Sample - Superstore.xls",   # .xls, .xlsx or .csv all work
    "sheet_name": "Orders",          # only used for Excel files
    "encoding": "latin-1",           # only used for CSV; use "utf-8" for most others
    "output_dir": "outputs",
    "test_months": 6,                # held-out months for model comparison
    "forecast_months": 6,            # future months to forecast
    # map YOUR column names -> standard names (left = your file, right = standard)
    "columns": {
        "Order Date": "date",
        "Product Name": "product",
        "Sub-Category": "subcategory",
        "Category": "category",
        "Region": "region",
        "Sales": "sales",
        "Quantity": "quantity",
        "Discount": "discount",
        "Profit": "profit",
    },
}
# ====================================================================

OUT = CONFIG["output_dir"]
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)


# ----------------------------- 1. LOAD & CLEAN -----------------------------
def load_and_clean() -> pd.DataFrame:
    path = CONFIG["data_path"]
    if path.lower().endswith((".xls", ".xlsx")):
        # Superstore Excel file has several sheets; the sales data is on "Orders"
        df = pd.read_excel(path, sheet_name=CONFIG["sheet_name"])
    else:
        df = pd.read_csv(path, encoding=CONFIG["encoding"])
    df = df.rename(columns=CONFIG["columns"])
    required = ["date", "sales"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. Edit CONFIG['columns'].")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "sales"]).drop_duplicates()

    # fill optional columns if absent so later steps don't break
    for col, default in [("product", "Unknown"), ("subcategory", "Unknown"), ("category", "Unknown"),
                         ("region", "Unknown"), ("quantity", 1), ("discount", 0.0), ("profit", np.nan)]:
        if col not in df.columns:
            df[col] = default

    for col in ["sales", "quantity", "discount", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quantity"] = df["quantity"].fillna(1)
    df["discount"] = df["discount"].fillna(0)
    df = df[df["sales"] > 0].copy()

    # time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["profit_margin"] = df["profit"] / df["sales"]

    df.to_csv(os.path.join(OUT, "sales_clean.csv"), index=False)
    print(f"Clean rows: {len(df):,} | {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


# ----------------------------- 2. EDA -----------------------------
def eda(df: pd.DataFrame) -> pd.Series:
    monthly = df.groupby("month_start")["sales"].sum().asfreq("MS").fillna(0)
    monthly.to_csv(os.path.join(OUT, "monthly_sales.csv"), header=["sales"])

    # monthly trend
    plt.figure(figsize=(11, 4))
    monthly.plot(marker="o")
    plt.title("Monthly Sales"); plt.ylabel("Sales"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "monthly_sales.png"), dpi=130); plt.close()

    # sales by category / region
    for col in ["category", "region"]:
        if df[col].nunique() > 1:
            g = df.groupby(col)[["sales", "profit"]].sum().sort_values("sales", ascending=False)
            g.plot(kind="bar", figsize=(8, 4)); plt.title(f"Sales & Profit by {col.title()}")
            plt.tight_layout(); plt.savefig(os.path.join(CHARTS, f"by_{col}.png"), dpi=130); plt.close()
    return monthly


# ----------------------------- 3. PROFITABILITY -----------------------------
def profitability(df: pd.DataFrame):
    if df["profit"].isna().all():
        print("No profit column - skipping profitability analysis.")
        return
    prod = (df.groupby(["category", "subcategory", "product"])
              .agg(sales=("sales", "sum"), profit=("profit", "sum"),
                   quantity=("quantity", "sum"), avg_discount=("discount", "mean"),
                   orders=("sales", "count"))
              .reset_index())
    prod["profit_margin"] = prod["profit"] / prod["sales"]
    prod["is_loss_making"] = prod["profit"] < 0
    prod = prod.sort_values("profit", ascending=False)
    prod.to_csv(os.path.join(OUT, "product_profitability.csv"), index=False)

    print("\nTop 10 products by profit:\n", prod.head(10)[["product", "sales", "profit", "profit_margin"]].to_string(index=False))
    print("\nBottom 10 products by profit:\n", prod.tail(10)[["product", "sales", "profit", "profit_margin"]].to_string(index=False))

    # high-sales but loss-making (candidates to re-price / drop)
    hi = prod[prod["sales"] > prod["sales"].median()]
    hi_loss = hi[hi["profit"] < 0].sort_values("profit")
    hi_loss.to_csv(os.path.join(OUT, "high_sales_loss_making.csv"), index=False)

    # discount effect on margin
    bins = [-0.001, 0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ["0%", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    df = df.copy()
    df["discount_band"] = pd.cut(df["discount"], bins=bins, labels=labels)
    d = df.groupby("discount_band", observed=True).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    d["margin"] = d["profit"] / d["sales"]
    d.to_csv(os.path.join(OUT, "discount_vs_margin.csv"))
    d["margin"].plot(kind="bar", figsize=(7, 4), color="tab:purple")
    plt.title("Profit Margin by Discount Band"); plt.axhline(0, color="k", lw=.8)
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "discount_vs_margin.png"), dpi=130); plt.close()

    # sales vs profit scatter (sub-category)
    sc = df.groupby("subcategory")[["sales", "profit"]].sum()
    plt.figure(figsize=(7, 5)); plt.scatter(sc["sales"], sc["profit"])
    for name, r in sc.iterrows():
        plt.annotate(name, (r["sales"], r["profit"]), fontsize=7)
    plt.axhline(0, color="r", lw=.8); plt.xlabel("Sales"); plt.ylabel("Profit")
    plt.title("Sales vs Profit by Sub-Category")
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "sales_vs_profit.png"), dpi=130); plt.close()


# ----------------------------- 4. SEASONALITY -----------------------------
def seasonality(df: pd.DataFrame, monthly: pd.Series):
    # month x year heatmap
    pivot = df.pivot_table(index="year", columns="month", values="sales", aggfunc="sum")
    pivot.to_csv(os.path.join(OUT, "seasonality_month_year.csv"))
    plt.figure(figsize=(11, 4)); plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.xticks(range(pivot.shape[1]), pivot.columns); plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.colorbar(label="Sales"); plt.title("Sales Heatmap (Year x Month)")
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "seasonality_heatmap.png"), dpi=130); plt.close()

    # seasonal index: avg month / overall avg month
    idx = (df.groupby(["year", "month"])["sales"].sum().groupby("month").mean())
    idx = (idx / idx.mean()).rename("seasonal_index")
    idx.to_csv(os.path.join(OUT, "seasonal_index.csv"))
    print("\nPeak months:", list(idx.sort_values(ascending=False).head(3).index),
          "| Weakest months:", list(idx.sort_values().head(3).index))

    # day-of-week pattern
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_of_week")["sales"].sum().reindex(order)
    dow.to_csv(os.path.join(OUT, "day_of_week_sales.csv"), header=["sales"])

    # decomposition
    if len(monthly) >= 24:
        res = seasonal_decompose(monthly, model="additive", period=12)
        fig = res.plot(); fig.set_size_inches(10, 7)
        plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "decomposition.png"), dpi=130); plt.close()


# ----------------------------- 5. FORECASTING -----------------------------
def seasonal_naive(train, h):
    last = train.iloc[-12:].values
    return np.array([last[i % 12] for i in range(h)])


def holt_winters(train, h):
    m = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    return m.forecast(h).values


def sarima(train, h):
    m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m.forecast(h).values


def prophet_model(train, h):
    from prophet import Prophet
    d = train.reset_index(); d.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(d)
    fut = m.make_future_dataframe(periods=h, freq="MS")
    return m.predict(fut)["yhat"].tail(h).values


def _make_ml():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9)
    except ImportError:
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, verbose=-1)
        except ImportError:
            return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)


def ml_model(train, h):
    """Lag/rolling-feature regressor with recursive multi-step forecasting."""
    df = pd.DataFrame({"y": train})
    for l in (1, 2, 3, 12):
        df[f"lag{l}"] = train.shift(l)
    df["roll3"] = train.shift(1).rolling(3).mean()
    df["month"] = df.index.month
    df = df.dropna()
    X, y = df.drop(columns="y"), df["y"]
    model = _make_ml().fit(X, y)

    hist = train.copy()
    preds = []
    for d in pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS"):
        row = pd.DataFrame([{
            "lag1": hist.iloc[-1], "lag2": hist.iloc[-2], "lag3": hist.iloc[-3],
            "lag12": hist.iloc[-12], "roll3": hist.iloc[-3:].mean(), "month": d.month}])[X.columns]
        p = float(model.predict(row)[0])
        preds.append(p)
        hist.loc[d] = p
    return np.array(preds)


def metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100
    return mae, rmse, mape


def forecasting(monthly: pd.Series):
    h_test, h_future = CONFIG["test_months"], CONFIG["forecast_months"]
    if len(monthly) < 24 + h_test:
        print(f"\nOnly {len(monthly)} months of data - need ~{24 + h_test}+ for seasonal models. Skipping forecasting.")
        return
    train, test = monthly.iloc[:-h_test], monthly.iloc[-h_test:]

    models = {"Seasonal Naive": seasonal_naive, "Holt-Winters": holt_winters,
              "SARIMA": sarima, "ML (lag features)": ml_model}
    try:
        import prophet  # noqa: F401
        models["Prophet"] = prophet_model
    except ImportError:
        print("Prophet not installed - skipping (pip install prophet).")

    rows, test_preds = [], {}
    for name, fn in models.items():
        try:
            p = fn(train, h_test)
            mae, rmse, mape = metrics(test.values, p)
            rows.append({"model": name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})
            test_preds[name] = p
        except Exception as e:
            print(f"{name} failed: {e}")
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)
    print("\nModel comparison (held-out last", h_test, "months):\n", res.round(2).to_string(index=False))

    best = res.loc[0, "model"]
    best_rmse = res.loc[0, "RMSE"]
    print(f"\nBest model: {best}")

    # refit on full history, forecast the future
    fut_pred = models[best](monthly, h_future)
    fut_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=h_future, freq="MS")
    fut_pred = np.maximum(fut_pred, 0)
    # approximate 95% interval from held-out RMSE (widens with horizon)
    width = 1.96 * best_rmse * np.sqrt(np.arange(1, h_future + 1) / h_future * 0.5 + 0.5)

    out = pd.concat([
        pd.DataFrame({"date": monthly.index, "actual": monthly.values, "forecast": np.nan,
                      "lower": np.nan, "upper": np.nan, "type": "Actual"}),
        pd.DataFrame({"date": fut_idx, "actual": np.nan, "forecast": fut_pred,
                      "lower": np.maximum(fut_pred - width, 0), "upper": fut_pred + width, "type": "Forecast"}),
    ])
    out["model"] = best
    out.to_csv(os.path.join(OUT, "forecast.csv"), index=False)

    plt.figure(figsize=(11, 4.5))
    plt.plot(monthly.index, monthly.values, label="Actual")
    plt.plot(fut_idx, fut_pred, "r--o", label=f"Forecast ({best})")
    plt.fill_between(fut_idx, np.maximum(fut_pred - width, 0), fut_pred + width, color="r", alpha=.15, label="95% interval")
    plt.title("Sales Forecast"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, "forecast.png"), dpi=130); plt.close()
    print("\nFuture forecast:\n", out[out.type == "Forecast"][["date", "forecast", "lower", "upper"]].round(0).to_string(index=False))


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    data = load_and_clean()
    monthly_sales = eda(data)
    profitability(data)
    seasonality(data, monthly_sales)
    forecasting(monthly_sales)
    print(f"\nDone. All outputs are in ./{OUT}/ (CSV files for Power BI + charts/ folder).")
