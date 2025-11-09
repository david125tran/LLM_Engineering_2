# CrewAI Regression Analysis Results

## Execution Summary

```
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Train the model
trained_model = RandomForestRegressor(random_state=42)
trained_model.fit(X_train, y_train)

# Make predictions
predictions = trained_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Feature importances
feature_importances = pd.Series(trained_model.feature_importances_, index=X_train.columns)

# Top 10 features
top_10_features = feature_importances.sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances:")
print(top_10_features)
```