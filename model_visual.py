import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import probplot
import settings
import model as m_lib


# Предсказания модели
X_train, X_test, y_train, y_test = m_lib.get_data_to_train_and_test()
model = m_lib.get_model()
y_pred = model.predict(X_test)

# 1. Важность признаков (коэффициенты для линейной регрессии)
if hasattr(model, 'coef_'):
    feature_importance = model.coef_
    feature_names = settings.FEATURE_COLUMNS

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.show()

# 2. Предсказанные vs фактические значения
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

# 3. Гистограмма остатков
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residual Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. QQ-plot для остатков
plt.figure(figsize=(8, 8))
probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-Plot of Residuals')
plt.grid(True)
plt.show()

# 5. Метрики качества
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f'Mean Absolute Error: {mae:.2f}')
print(f"R^2 Score: {r2:.2f}")
