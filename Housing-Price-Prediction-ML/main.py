# -----------------------------
# 1. Data Preprocessing
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('E:\DELL\Desktop\Housing-Price-Prediction-ML\Housing.csv')

# Encode categorical features
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Train-test split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# -----------------------------
# 3. Model Training
# -----------------------------
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from time import time

models = {
    'LinearRegression': LinearRegression(),
    'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, learning_rate='constant'),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=5000)  # Increased max_iter to reduce convergence warning
}

results = []
for name, model in models.items():
    start = time()
    model.fit(X_train_poly, y_train)
    train_time = time() - start
    preds = model.predict(X_test_poly)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    results.append([name, rmse, r2, train_time])

results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R2 Score', 'Training Time (s)'])

# -----------------------------
# 4. Learning Curve Visualization
# -----------------------------
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, title):
    train_sizes, train_scores, val_scores = learning_curve(model, X_train_poly, y_train,
                                                           cv=5, scoring='neg_mean_squared_error',
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    plt.plot(train_sizes, train_rmse, label='Train RMSE')
    plt.plot(train_sizes, val_rmse, label='Val RMSE')
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example: plot_learning_curve(models['Ridge'], 'Ridge Regression Learning Curve')

# -----------------------------
# 5. Hyperparameter Tuning
# -----------------------------
from sklearn.model_selection import GridSearchCV

param_grid_ridge = {'alpha': [0.01, 0.1, 1.0, 10]}
param_grid_lasso = {'alpha': [0.01, 0.1, 1.0, 10]}

grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, scoring='r2', cv=5)
grid_lasso = GridSearchCV(Lasso(max_iter=5000), param_grid_lasso, scoring='r2', cv=5)
grid_ridge.fit(X_train_poly, y_train)
grid_lasso.fit(X_train_poly, y_train)

# -----------------------------
# 6. Coefficient Analysis Plot
# -----------------------------
coefs = pd.Series(grid_ridge.best_estimator_.coef_, index=poly.get_feature_names_out(X.columns))
coefs.sort_values().plot(kind='barh', figsize=(8,12))
plt.title('Feature Importances from Ridge')
plt.show()

# -----------------------------
# 7. Logistic Regression (Classification)
# -----------------------------
# Convert to classification: binarize price into high/low price based on median
y_class = (y > y.median()).astype(int)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_clf = scaler.fit_transform(X_train_clf)
X_test_clf = scaler.transform(X_test_clf)

log_reg = LogisticRegression()
log_reg.fit(X_train_clf, y_train_clf)
acc = log_reg.score(X_test_clf, y_test_clf)

from sklearn.metrics import f1_score
f1 = f1_score(y_test_clf, log_reg.predict(X_test_clf))

# -----------------------------
# 8. Report Summary (Printed)
# -----------------------------
print("\nModel Comparison:")
print(results_df)
print("\nBest Ridge Alpha:", grid_ridge.best_params_)
print("Best Lasso Alpha:", grid_lasso.best_params_)
print("\nLogistic Regression Accuracy:", acc)
print("F1 Score:", f1)

# -----------------------------
# 9. Visualization of Model Results
# -----------------------------
plt.figure(figsize=(10, 6))
bar = plt.bar(results_df['Model'], results_df['RMSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Model RMSE Comparison')
for rect in bar:
    y_val = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, y_val, round(y_val, 2), ha='center', va='bottom')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
bar = plt.bar(results_df['Model'], results_df['R2 Score'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.title('Model R² Score Comparison')
for rect in bar:
    y_val = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, y_val, round(y_val, 2), ha='center', va='bottom')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
