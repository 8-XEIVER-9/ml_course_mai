# Шпаргалка по коду (лабы 2 и 3)

Всё на `pandas` + `scikit-learn`. Порядок блоков = порядок, в котором пишется лаба.

---

## 1. Загрузка и первый взгляд на данные

```python
import pandas as pd
df = pd.read_csv("data/Titanic-Dataset.csv")

df.shape                 # (строк, столбцов)
df.head()                # первые 5 строк
df.info()                # типы столбцов и сколько НЕ пустых значений
df.describe()            # среднее, мин, макс и т.д. для чисел
df.isna().sum()          # сколько пропусков в каждом столбце
df.duplicated().sum()    # сколько полностью одинаковых строк
df["Name"].nunique()     # сколько разных значений в столбце
df["Survived"].value_counts(normalize=True)    # доли классов (normalize=True -> доли, а не штуки)
df.groupby("Sex")["Survived"].mean()           # доля выживших по полу
```

## 2. Графики

```python
import matplotlib.pyplot as plt
import seaborn as sns

df["Survived"].value_counts().plot(kind="bar")              # столбики
sns.barplot(data=df, x="Sex", y="Survived")                 # доля выживших по группам
sns.histplot(data=df, x="Age", hue="Survived", bins=30)     # гистограмма, раскраска по классу
series.sort_values().plot(kind="barh")                      # горизонтальные столбики (для важности признаков)
plt.title("..."); plt.show()
```

## 3. Признаки X и целевая y

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1        # новый признак из старых

num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
cat_features = ["Sex", "Embarked"]

X = df[num_features + cat_features]     # ID, имя, билет, каюту НЕ берём
y = df["Survived"]                      # что предсказываем
```

## 4. Разбиение train / test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% в тест
    random_state=42,      # фиксируем случайность -> одинаковый результат каждый раз
    stratify=y,           # одинаковая доля классов в train и test
)
```

## 5. Предобработка: пропуски, категории, масштаб

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),         # пропуск -> медиана
        ("scaler", StandardScaler()),                          # среднее 0, разброс 1
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # пропуск -> самое частое
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),  # текст -> столбцы 0/1
    ]), cat_features),
], verbose_feature_names_out=False)
```

| Инструмент | Что делает |
|---|---|
| `SimpleImputer(strategy="median")` | заполняет пропуски медианой (`"mean"` — средним, `"most_frequent"` — самым частым) |
| `StandardScaler()` | `(x - среднее) / std` → все признаки в одном масштабе |
| `OneHotEncoder()` | `Embarked = S/C/Q` → три столбца `Embarked_S`, `Embarked_C`, `Embarked_Q` из 0 и 1 |
| `ColumnTransformer` | разным столбцам — разная обработка |
| `Pipeline` | склеивает шаги в цепочку: `fit` обучает всё сразу, и только на train |

**Почему Pipeline важен:** медиана и среднее для масштабирования считаются **только по train**. Если сначала обработать весь `df`, а потом разбить — информация из test протечёт в обучение (ошибка «утечка данных»).

```python
def make_model(clf):
    return Pipeline([("prep", preprocess), ("model", clf)])

model = make_model(RandomForestClassifier())
model.fit(X_train, y_train)       # обучение (и предобработки, и модели)
pred = model.predict(X_test)      # предсказание 0/1
```

## 6. Модели классификации

```python
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

DummyClassifier(strategy="most_frequent")                 # baseline: всегда самый частый класс
DecisionTreeClassifier(max_depth=None, random_state=42)    # одно дерево
RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)   # лес из 300 деревьев
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
SVC()                                                      # метод опорных векторов (нужен масштаб!)
KNeighborsClassifier(n_neighbors=5)                        # 5 ближайших соседей (нужен масштаб!)
```

`n_jobs=-1` — использовать все ядра процессора (быстрее).

## 7. Метрики

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

accuracy_score(y_test, pred)
precision_score(y_test, pred, zero_division=0)   # zero_division=0 — чтобы не было ошибки, если модель ни разу не сказала "1"
recall_score(y_test, pred)
f1_score(y_test, pred)
print(classification_report(y_test, pred))       # всё сразу по обоим классам

confusion_matrix(y_test, pred)                   # [[TN, FP], [FN, TP]]
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["погиб", "выжил"], cmap="Blues")
plt.show()
```

Проверка переобучения: сравнить метрику на train и test.

```python
f1_score(y_train, model.predict(X_train))   # если тут ~1.0, а на test сильно меньше -> переобучение
```

## 8. Важность признаков (только у деревьев, леса, бустинга)

```python
names = model.named_steps["prep"].get_feature_names_out()          # имена после One-Hot
imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
imp.sort_values().plot(kind="barh"); plt.show()

# сравнить три модели на одном графике
pd.DataFrame({"DT": dt_imp, "RF": rf_imp, "GB": gb_imp}).plot(kind="barh")
```

`model.named_steps["model"]` — достать модель из Pipeline по имени шага.

## 9. Итоговая таблица (leaderboard)

```python
results = {}                                     # перед обучением моделей
results["SVM"] = {"Accuracy": ..., "F1": ...}    # после каждой модели

res = pd.DataFrame(results).T.sort_values("F1", ascending=False)   # .T — повернуть таблицу
res["F1 > 0.70"] = res["F1"] > 0.70
res.round(3)
```

## 10. Анализ ошибок

```python
errors = X_test.copy()
errors["true"] = y_test
errors["pred"] = model.predict(X_test)
errors["wrong"] = errors["true"] != errors["pred"]

errors.groupby("Sex")["wrong"].mean()      # доля ошибок по группам
pd.crosstab(errors["Sex"], errors["wrong"])  # таблица "группа x ошибся/нет"
errors[errors["wrong"]].join(df[["Name"]]).head(10)   # посмотреть ошибочных пассажиров
```

---

# Лаба 3: подбор гиперпараметров

## 11. Кросс-валидация

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # 5 частей, доля классов сохраняется
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")   # ТОЛЬКО train!
scores            # 5 чисел — F1 на каждом fold
scores.mean()     # среднее качество
scores.std()      # разброс (чем меньше, тем стабильнее модель)
```

`scoring` — какая метрика: `"f1"`, `"accuracy"`, `"roc_auc"`, `"neg_root_mean_squared_error"` (для регрессии, со знаком минус).

## 12. Grid Search — перебор всех комбинаций

```python
from sklearn.model_selection import GridSearchCV
import time

param_grid = {
    "model__n_estimators": [50, 100, 200, 400],      # "model__" = имя шага в Pipeline + два подчёркивания
    "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
    "model__max_depth": [1, 2, 3, 4, 5],
}                                                   # 4*4*5 = 80 комбинаций

grid = GridSearchCV(make_model(GradientBoostingClassifier(random_state=42)), param_grid,
                    cv=cv, scoring="f1", n_jobs=-1, return_train_score=True)
start = time.time()
grid.fit(X_train, y_train)
print(time.time() - start, "сек")

grid.best_params_       # лучшие параметры
grid.best_score_        # лучший средний F1 на CV
grid.best_estimator_    # готовая модель с лучшими параметрами (уже переобучена на всём train)
pd.DataFrame(grid.cv_results_)    # таблица по всем комбинациям
```

## 13. Random Search — случайные комбинации

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

param_dist = {
    "model__n_estimators": randint(50, 501),         # случайное целое от 50 до 500
    "model__learning_rate": loguniform(0.01, 0.3),   # случайное дробное, чаще маленькие
    "model__max_depth": randint(1, 6),               # от 1 до 5
}
rand = RandomizedSearchCV(model, param_dist, n_iter=40,   # n_iter — сколько комбинаций попробовать
                          cv=cv, scoring="f1", n_jobs=-1, random_state=42)
rand.fit(X_train, y_train)
rand.best_params_, rand.best_score_
```

## 14. Влияние одного параметра (validation curve)

```python
from sklearn.model_selection import validation_curve

values = [1, 2, 3, 4, 5, 6, 8, 10]
train_sc, val_sc = validation_curve(model, X_train, y_train,
                                    param_name="model__max_depth", param_range=values,
                                    cv=cv, scoring="f1", n_jobs=-1)
plt.plot(values, train_sc.mean(axis=1), "o-", label="train")
plt.plot(values, val_sc.mean(axis=1), "o-", label="CV")
plt.legend(); plt.show()
```

Как читать: обе линии низко → **недообучение**; train высоко, CV падает → **переобучение**; CV на максимуме → **хорошо**.

## 15. Результаты по отдельным folds

```python
i = grid.best_index_
[grid.cv_results_[f"split{k}_test_score"][i] for k in range(5)]   # F1 лучшей комбинации на каждом fold
```

## 16. Финал — один раз на test

```python
best = grid.best_estimator_
best.fit(X_train, y_train)
f1_score(y_test, best.predict(X_test))   # смотрим test ТОЛЬКО здесь, в самом конце
```

---

## Частые ошибки

| Ошибка | Как правильно |
|---|---|
| `fit` скейлера/импьютера на всём `df` | всё в `Pipeline`, `fit` только на `X_train` |
| подбирать параметры, глядя на test | подбирать по `cross_val_score` / `GridSearchCV` на train |
| забыть `random_state` | ставить везде: split, модели, CV, поиск |
| в параметрах сетки нет `model__` | если модель в Pipeline, имя параметра = `<шаг>__<параметр>` |
| KNN/SVM без масштабирования | всегда `StandardScaler` |
| ID, имена, номера билетов в признаках | выкинуть — модель их «запомнит» |
