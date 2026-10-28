
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ==========================================
# 1. السلع والأسعار الجديدة
# ==========================================

products = {
    "Flour": 25,
    "Rice": 18,
    "Oil": 30,
    "Sugar": 15,
    "Milk": 12,
    "Fuel": 35,
    "Honey": 45,
    "Dates": 28,
    "Beans": 16,
    "Lentils": 18
}

rows = []

# إنشاء بيانات تجريبية لمدة 180 يومًا
for product, base_price in products.items():
    for day in range(1, 181):

        price = base_price * (1 + 0.0015 * day)
        price += np.random.normal(0, base_price * 0.04)

        availability = np.random.randint(5, 101)
        demand = np.random.randint(20, 151)

        rows.append({
            "Day": day,
            "Product": product,
            "Price": max(0.1, round(price, 2)),
            "Availability": availability,
            "Demand": demand
        })

df = pd.DataFrame(rows)

# إضافة حالات ارتفاع غير طبيعي في الأسعار
anomaly_indices = np.random.choice(
    df.index, size=20, replace=False
)

df.loc[anomaly_indices, "Price"] *= np.random.uniform(
    1.8, 2.5, size=len(anomaly_indices)
)

df["Price"] = df["Price"].round(2)

print("=" * 60)
print("MARKET CRISIS ML - MARKET ANALYSIS")
print("=" * 60)

print("\nFirst 10 Records:")
print(df.head(10).to_string(index=False))

print("\nTotal Records:", len(df))
print("Total Products:", df["Product"].nunique())

# ==========================================
# 2. تحليل الأسعار
# ==========================================

print("\n" + "=" * 60)
print("PRICE STATISTICS")
print("=" * 60)

price_stats = df.groupby("Product")["Price"].agg(
    ["mean", "median", "min", "max", "std"]
).round(2)

print(price_stats)

# ==========================================
# 3. كشف الحالات الشاذة
# ==========================================

detector = IsolationForest(
    contamination=0.02,
    random_state=42
)

df["Anomaly"] = detector.fit_predict(
    df[["Price", "Availability", "Demand"]]
)

df["Anomaly"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

anomalies = df[df["Anomaly"] == "Anomaly"]

print("\n" + "=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

print("Anomalies Detected:", len(anomalies))
print("Normal Records:", len(df) - len(anomalies))

print("\nAnomaly Samples:")
print(anomalies.head(10).to_string(index=False))

# ==========================================
# 4. تجميع السلع باستخدام KMeans
# ==========================================

features = df.groupby("Product").agg(
    Average_Price=("Price", "mean"),
    Average_Availability=("Availability", "mean"),
    Average_Demand=("Demand", "mean")
).round(2)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

features["Cluster"] = kmeans.fit_predict(scaled_features)

print("\n" + "=" * 60)
print("KMEANS PRODUCT CLUSTERING")
print("=" * 60)

print(features.sort_values(
    "Average_Price", ascending=False
))

# ==========================================
# 5. التنبؤ بالأسعار
# ==========================================

model_df = df.copy()

# ترميز أسماء السلع رقميًا
product_categories = pd.Categorical(
    model_df["Product"],
    categories=list(products.keys())
)

model_df["Product_Code"] = product_categories.codes

X = model_df[
    ["Day", "Product_Code", "Availability", "Demand"]
]

y = model_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("PRICE PREDICTION RESULTS")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"R2 Percentage: {r2 * 100:.2f}%")

# ==========================================
# 6. توقع أسعار اليوم التالي
# ==========================================

print("\n" + "=" * 60)
print("NEXT DAY PRICE PREDICTIONS")
print("=" * 60)

next_day = 181

for product, product_code in enumerate(products.keys()):

    history = df[df["Product"] == product_code].tail(7)

    next_input = pd.DataFrame([{
        "Day": next_day,
        "Product_Code": product,
        "Availability": history["Availability"].mean(),
        "Demand": history["Demand"].mean()
    }])

    predicted_price = model.predict(next_input)[0]

    print(f"{product_code}: {predicted_price:.2f}")

# ==========================================
# 7. رسم متوسط الأسعار
# ==========================================

plt.figure(figsize=(12, 6))

average_prices = df.groupby("Product")["Price"].mean()
average_prices = average_prices.sort_values(ascending=False)

average_prices.plot(kind="bar")

plt.title("Average Market Prices")
plt.xlabel("Product")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 8. رسم تغير الأسعار عبر الأيام
# ==========================================

plt.figure(figsize=(13, 7))

for product in products:
    product_data = df[df["Product"] == product]

    daily_prices = product_data.groupby("Day")["Price"].mean()

    plt.plot(
        daily_prices.index,
        daily_prices.values,
        label=product
    )

plt.title("Market Price Changes Over 180 Days")
plt.xlabel("Day")
plt.ylabel("Price")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# ==========================================
# 9. مقارنة الأسعار الحقيقية والمتوقعة
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.5)

min_price = min(y_test.min(), predictions.min())
max_price = max(y_test.max(), predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.show()

# ==========================================
# 10. حفظ النتائج
# ==========================================

df.to_csv(
    "market_data_results.csv",
    index=False,
    encoding="utf-8-sig"
)

features.to_csv(
    "market_clusters.csv",
    encoding="utf-8-sig"
)

anomalies.to_csv(
    "market_anomalies.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nResults saved successfully:")
print("1. market_data_results.csv")
print("2. market_clusters.csv")
print("3. market_anomalies.csv")

print("\nMarket Crisis ML Analysis Completed!")
