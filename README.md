import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

alpha_grid = np.logspace(-4, 4, 100)

lr = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alpha_grid, cv=5)).fit(X_train, y_train)
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alpha_grid, cv=5, max_iter=10000)).fit(X_train, y_train)

print(f"{'Model':10s} | {'MSE':>10s} | {'R2':>8s}")
print("-" * 36)

for name, model in [("Linear", lr), ("Ridge", ridge), ("Lasso", lasso)]:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:10s} | {mse:10.2f} | {r2:8.4f}")

print("\n" + "=" * 36)
print(f"Best Ridge alpha: {ridge.named_steps['ridgecv'].alpha_:.4f}")
print(f"Best Lasso alpha: {lasso.named_steps['lassocv'].alpha_:.4f}")
print("=" * 36)

plt.figure(figsize=(10, 5))
plt.plot(lr.named_steps['linearregression'].coef_, 'bo-', label='Linear')
plt.plot(ridge.named_steps['ridgecv'].coef_, 'rs-', label='Ridge')
plt.plot(lasso.named_steps['lassocv'].coef_, 'g^-', label='Lasso')
plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
plt.xlabel("Feature index")
plt.ylabel("Standardized Coefficient Value")
plt.title("Coefficient Comparison — Ridge vs Lasso vs Linear")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
