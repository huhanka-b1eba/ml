import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 0. Загрузка данных
df = pd.read_csv("AmesHousing.csv")

print(df.shape)
print(df.head())


#  1. Чистка пропусков
none_cols = [
    "Alley", "Pool QC", "Fence", "Misc Feature",
    "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
    "BsmtFin Type 1", "BsmtFin Type 2",
    "Mas Vnr Type"
]

for col in none_cols:
    if col in df.columns:
        df[col] = df[col].fillna("None")
        df[col] = df[col].replace("NA", "None")


zero_cols = [
    "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Area", "Garage Cars", "Garage Yr Blt",
    "Mas Vnr Area",
    "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF"
]

for col in zero_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)


# 3. Lot Frontage заполняем медианой по району
if "Lot Frontage" in df.columns:
    df["Lot Frontage"] = pd.to_numeric(df["Lot Frontage"], errors="coerce")
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())


# 2. остальные числовые пропуски медианой
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# остальные категориальные пропуски самым частым значением
cat_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 11. Новые признаки. Создать признак «возраст дома на момент продажи» и «лет с последнего ремонта».
df["House Age"] = df["Yr Sold"] - df["Year Built"]
df["Years Since Remodel"] = df["Yr Sold"] - df["Year Remod/Add"]

df["House Age"] = df["House Age"].clip(lower=0)
df["Years Since Remodel"] = df["Years Since Remodel"].clip(lower=0)


# One-Hot Encoding
y = df["SalePrice"]

X = df.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")

X_encoded = pd.get_dummies(X, drop_first=True)


# Функция оценки модели
def check_model(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=10)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    print("\n", title)
    print("MAE:", mean_absolute_error(y_test, preds)) # Средняя ошибка
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds))) # Корень из средней квадратичной ошибки
    print("R2:", r2_score(y_test, preds)) # Коэффициент детерминации показывает, какая доля дисперсии зависимой переменной объясняется моделью. Чем ближе к 1, тем лучше модель объясняет данные.

    return model, scaler, X_train.columns


ridge_model, scaler, feature_names = check_model(X_encoded, y, "Ridge ДО удаления аномалий")


# 4. Топ-10 признаков Ridge
coefs = pd.DataFrame({
    "feature": feature_names,
    "coef": ridge_model.coef_
})

coefs["abs_coef"] = coefs["coef"].abs()

top10 = coefs.sort_values("abs_coef", ascending=False).head(10)

print("\nТОП-10 важных признаков:")
print(top10[["feature", "coef"]])

plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["abs_coef"])
plt.gca().invert_yaxis()
plt.title("Топ-10 важных признаков Ridge")
plt.xlabel("Модуль коэффициента")
plt.tight_layout()
plt.show()


# 5. Цена от жилой площади
plt.figure(figsize=(8, 6))
plt.scatter(df["Gr Liv Area"], df["SalePrice"], alpha=0.6)
plt.xlabel("Gr Liv Area")
plt.ylabel("SalePrice")
plt.title("Цена от жилой площади")
plt.tight_layout()
plt.show()


# 6. Поиск аномалий
# подозрительно дешевые большие дома
df["price_per_area"] = df["SalePrice"] / df["Gr Liv Area"]

q1 = df["price_per_area"].quantile(0.25)
q3 = df["price_per_area"].quantile(0.75)
iqr = q3 - q1

low_border = q1 - 1.5 * iqr

area_border = df["Gr Liv Area"].quantile(0.90)

anomalies_iqr = df[
    (df["price_per_area"] < low_border) &
    (df["Gr Liv Area"] > area_border)
    ]

print("\nАномалии по IQR:")
print(anomalies_iqr[["Gr Liv Area", "SalePrice", "price_per_area"]])


# Isolation Forest
iso_data = df[["Gr Liv Area", "SalePrice"]]

iso = IsolationForest(contamination=0.03, random_state=42)
df["is_anomaly"] = iso.fit_predict(iso_data)

anomalies_iso = df[df["is_anomaly"] == -1]

print("\nАномалии по Isolation Forest:")
print(anomalies_iso[["Gr Liv Area", "SalePrice"]])


# Удалим аномалии по IQR
df_clean = df.drop(anomalies_iqr.index)

X_clean = df_clean.drop(
    columns=["SalePrice", "Order", "PID", "price_per_area", "is_anomaly"],
    errors="ignore"
)

y_clean = df_clean["SalePrice"]

X_clean_encoded = pd.get_dummies(X_clean, drop_first=True)

# выравниваем колонки с исходным X_encoded
X_clean_encoded = X_clean_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# 7. Сравнить качество регрессии «до» и «после» удаления аномалий.
check_model(X_clean_encoded, y_clean, "Ridge ПОСЛЕ удаления аномалий")


# 8. Сегментация домов без учета цены
segment_features = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Area",
    "House Age",
    "Years Since Remodel",
    "Full Bath",
    "Bedroom AbvGr"
]

segment_data = df[segment_features].copy()

scaler_seg = StandardScaler()
segment_scaled = scaler_seg.fit_transform(segment_data)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(segment_scaled)

print("\nСегменты домов:")
print(df.groupby("segment")[segment_features].mean())


# 10. PCA по числовым признакам
numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

numeric_df = numeric_df.drop(
    columns=["SalePrice", "price_per_area", "is_anomaly"],
    errors="ignore"
)

y_pca = df["SalePrice"]

scaler_pca = StandardScaler()
numeric_scaled = scaler_pca.fit_transform(numeric_df)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(numeric_scaled)

print("\nДоля объясненной дисперсии PCA:")
print(pca.explained_variance_ratio_)

check_model(pd.DataFrame(X_pca), y_pca, "Ridge на PCA-компонентах")

# 10. Анализ цен по годам и месяцам
price_by_year = df.groupby("Yr Sold")["SalePrice"].mean()
price_by_month = df.groupby("Mo Sold")["SalePrice"].mean()

print("\nСредняя цена по годам:")
print(price_by_year)

print("\nСредняя цена по месяцам:")
print(price_by_month)


plt.figure(figsize=(8, 5))
price_by_year.plot(marker="o")
plt.title("Средняя цена по годам")
plt.xlabel("Год продажи")
plt.ylabel("Средняя цена")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
price_by_month.plot(marker="o")
plt.title("Средняя цена по месяцам")
plt.xlabel("Месяц продажи")
plt.ylabel("Средняя цена")
plt.tight_layout()
plt.show()

# 12. Сравнение до/после кризиса
before_2008 = df[df["Yr Sold"] < 2008]["SalePrice"].mean()
during_after_2008 = df[df["Yr Sold"] >= 2008]["SalePrice"].mean()

drop_percent = (before_2008 - during_after_2008) / before_2008 * 100

print("\nСредняя цена до 2008:", before_2008)
print("Средняя цена с 2008 и позже:", during_after_2008)
print("Изменение в процентах:", drop_percent)

# Сезонность: весна против зимы
spring = df[df["Mo Sold"].isin([3, 4, 5])]["SalePrice"].mean()
winter = df[df["Mo Sold"].isin([12, 1, 2])]["SalePrice"].mean()

season_diff = (spring - winter) / winter * 100

print("\nСредняя цена весной:", spring)
print("Средняя цена зимой:", winter)
print("Разница весна vs зима в процентах:", season_diff)

# Удаление аномальных объектов улучшило качество модели.
# Выбросы мешали регрессии и ухудшали предсказания.

# применение PCA уменьшило размерность, но ухудшило качество регрессии.

# после кризиса 2008 средняя стоимость снизилась примерно на 2.7%.

# зимой средняя стоимость оказалась выше примерно на 7%.
