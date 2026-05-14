import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Загрузка данных
df = pd.read_csv("heart.csv")

print(df.shape)
print(df.head())

print("\nБаланс классов:")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))


# X и y
X = df.drop(columns=["target"])
y = df["target"]

feature_names = X.columns


# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Функция проверки модели
def check_classification_model(model, X_test, y_test, title):
    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)

    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    print("Accuracy:", accuracy_score(y_test, preds)) # Доля правильных ответов
    print("Precision:", precision_score(y_test, preds)) # Доля правильных положительных ответов среди всех, которые модель предсказала как положительные. Высокая точность означает, что модель редко ошибается, предсказывая положительный класс.
    print("Recall:", recall_score(y_test, preds)) # Доля правильных положительных ответов среди всех реальных положительных примеров. Высокая полнота означает, что модель редко пропускает положительные примеры.
    print("F1-score:", f1_score(y_test, preds)) # Гармоническое среднее между точностью и полнотой. Высокий F1-score означает, что модель хорошо балансирует между точностью и полнотой.
    print("ROC-AUC:", roc_auc_score(y_test, probs)) # Площадь под ROC-кривой. Чем ближе к 1, тем лучше модель различает классы.

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification report:")
    print(classification_report(y_test, preds))


# Модели по умолчанию
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

svc_linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear", probability=True))
])

tree_model = DecisionTreeClassifier(random_state=42)


log_model.fit(X_train, y_train)
svc_linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

check_classification_model(log_model, X_test, y_test, "Logistic Regression DEFAULT")
check_classification_model(svc_linear_model, X_test, y_test, "SVC Linear DEFAULT")
check_classification_model(tree_model, X_test, y_test, "Decision Tree DEFAULT")


# 2. GridSearchCV
log_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear", "saga"]
}

svc_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto", 0.01, 0.1],
    "model__kernel": ["rbf", "poly"]
}

tree_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}


log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True))
])


log_search = GridSearchCV(
    log_pipe,
    log_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

svc_search = GridSearchCV(
    svc_pipe,
    svc_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

tree_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    tree_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


log_search.fit(X_train, y_train)
svc_search.fit(X_train, y_train)
tree_search.fit(X_train, y_train)


print("\nЛучшие параметры LogisticRegression:")
print(log_search.best_params_)

print("\nЛучшие параметры SVC:")
print(svc_search.best_params_)

print("\nЛучшие параметры DecisionTree:")
print(tree_search.best_params_)


best_log = log_search.best_estimator_
best_svc = svc_search.best_estimator_
best_tree = tree_search.best_estimator_


check_classification_model(best_log, X_test, y_test, "Logistic Regression AFTER GridSearch")
check_classification_model(best_svc, X_test, y_test, "SVC AFTER GridSearch")
check_classification_model(best_tree, X_test, y_test, "Decision Tree AFTER GridSearch")


# 3. Коэффициенты логистической регрессии
log_reg = best_log.named_steps["model"]

coefs = pd.DataFrame({
    "feature": feature_names,
    "coef": log_reg.coef_[0]
})

coefs["abs_coef"] = coefs["coef"].abs()

top10 = coefs.sort_values("abs_coef", ascending=False).head(10)

print("\nТОП-10 признаков Logistic Regression:")
print(top10[["feature", "coef"]])


plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["abs_coef"])
plt.gca().invert_yaxis()
plt.title("Топ-10 важных признаков Logistic Regression")
plt.xlabel("Модуль коэффициента")
plt.tight_layout()
plt.show()


# 4. Визуализация дерева решений
plt.figure(figsize=(20, 10))

plot_tree(
    best_tree,
    feature_names=feature_names,
    class_names=["No disease", "Disease"],
    filled=True,
    max_depth=4,
    fontsize=9
)

plt.title("Decision Tree max_depth=4")
plt.show()


# Какой признак в корне дерева
root_feature_index = best_tree.tree_.feature[0]
root_feature_name = feature_names[root_feature_index]

print("\nПризнак в корне дерева:")
print(root_feature_name)


# 5. Веса SVM с линейным ядром
linear_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear"))
])

linear_svc.fit(X_train, y_train)

svm_coefs = pd.DataFrame({
    "feature": feature_names,
    "coef": linear_svc.named_steps["model"].coef_[0]
})

svm_coefs["abs_coef"] = svm_coefs["coef"].abs()

print("\nТОП-10 признаков Linear SVM:")
print(
    svm_coefs
    .sort_values("abs_coef", ascending=False)
    .head(10)[["feature", "coef"]]
)

# В корне дерева решений находится признак cp, он также оказался самым важным признаком в логистической регрессии