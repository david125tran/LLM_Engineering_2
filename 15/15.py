# ---------------------------------- Libraries ----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # Added for missing value handling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings 
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")



# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualization of what's going on
    """
    banner_len = len(text)
    mid = 49 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 50)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 50)



# ---------------------------------- Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)



# ---------------------------------- Load & Inspect Data ----------------------------------
print_banner("Load Data")

# Configure settings for better visualization
sns.set(style ="whitegrid")  # Set a nice default style for seaborn plots
plt.rcParams["figure.figsize"] = (12, 6)  # Set default figure size for matplotlib plots
print("Libraries Imported Successfully!")

file_path = script_dir + r"\\Supplement_Sales_Weekly.csv"
df = pd.read_csv(file_path)
print(f"\nDataset loaded successfully from {file_path}")

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nFirst five rows:")

print(df.head())

print("\nDataset information:")
print(df.info)

print("\nDescriptive Statistics (Numerical Columns):")
print(df.describe())

print("\nDescriptive Statistics (Categorical Columns):")
print(df.describe(include = ["object"]))

print("\nMissing values:")
print(df.isnull())

print("\nInitial check for missing values:")
print(df.isnull().sum())



# ---------------------------------- Handling Missing Values ----------------------------------
print_banner("Handling Missing Values")

# Ways to handle missing values:

# * Dropping:               Remove rows or columns with missing values. This is simple, but can lead to significant data loss if many 
#                           values are missing. Generally not recommended unless the proportion of missing data is tiny or a column is 
#                           mostly empty.
# * Imputation:             Replace missing values with estimated or calculated values. Common imputation techniques include:
#     a) Mean Imputation:   Replace missing numerical values with the mean of the column. Sensitive to outliers.
#     b) Median Imputation: Replace missing numerical values with the median of the column. More robust to outliers than the mean.
#     c) Mode Imputation:   Replace missing categorical values with the mode (most frequent value) of the column.
#     d) More Advanced:     Regression imputation, K-Nearest Neighbors (KNN) imputation, etc. (beyond the scope of this project). For our 
#                           numerical columns Price and Discount, median imputation is often a good starting point as it's less affected
#                           by unusually high or low values (outliers). We'll use scikit-learn's SimpleImputer.

strategy = "mean"
# strategy = "median"
# strategy = "most_frequent"
# strategy = "constant"

print(f"Handling missing values using '{strategy}' Imputation...")

# Identify columns with missing values (should be Price and Discount now)
cols_with_missing = ["Price", "Discount"]

# Initialize the SimpleImputer with the {strategy} strategy
imputer = SimpleImputer(strategy=strategy)

# Apply the imputer to the selected columns
# .fit_transform() calculates the value and replaces NaNs in one step
# Important: It returns a NumPy array, so we need to put it back into the DataFrame
df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])

print(f"\nMissing values after {strategy} Imputation:")
print(df.isnull().sum())

if df[cols_with_missing].isnull().sum().sum() == 0:
    print("\nMissing values in 'Price' and 'Discount' successfully imputed.")
else:
    print("\nWarning: Missing values still detected after imputation attempt.")



# ---------------------------------- Exploratory Data Analysis ----------------------------------
print_banner("Exploratory Data Analysis")

# Show plots
histplot                =   False
histplot_and_boxplot    =   False
countplot               =   False
scatterplot             =   False
heatmap                 =   False
boxplot                 =   False

if histplot:
    # Analyze the distribution of the target variable (Units Sold)
    # If skewness is high (e.g., > 1 or < -1), a log transform might be considered later.
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Units Sold"], kde = True, bins = 30)
    plt.title("Distribution of Units Sold")
    plt.xlabel("Units Sold")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Units Sold Skewness: {df['Units Sold'].skew():.2f}")

if histplot_and_boxplot:
    numerical_features = [ "Price", "Discount"]  # These should now be imputed
    print("\nAnalyzing distributions of key numerical features (after imputation):")

    # Plot the histogram and boxplot for each numerical feature
    for col in numerical_features:
        plt.figure(figsize = (10, 4))
        sns.histplot(df[col], kde = True, bins = 25)
        plt.title(f"Distribution of {col}")
        plt.show()

        plt.figure(figsize = (10, 2))
        sns.boxplot(x = df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()
        print("-" * 130)

if countplot:
    # Analyze categorical features (Category, Location, Platform)
    categorical_features = ["Category", "Location", "Platform"]
    print("\nAnalyzing distributions of categorical features:")

    # Show counts in each category 
    for col in categorical_features:
        plt.figure(figsize = (12, 5))
        sns.countplot(y = df[col], order = df[col].value_counts().index, palette = "viridis")  # Use y for horizontal bars if many categories
        plt.title(f"Frequency of each {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.show()
        print("-" * 130)

if scatterplot:
    # View the scatterplot between Numerical vs. Target (Units Sold)
    print("\nAnalyzing relationships between numerical features and Revenue:")
    numerical_predictors = ["Price", "Discount"]

    for col in numerical_predictors:
        plt.figure(figsize = (8, 5))
        sns.scatterplot(x = df[col], y = df["Units Sold"], alpha = 0.5)  # Added alpha for density
        plt.title(f"{col} vs Units Sold")
        plt.show()

if heatmap:
    # View the correlation heatmap (only for numerical columns)
    print("\nCorrelation Matrix Heatmap (Numerical Features including Revenue):")

    # Only select numeric types for correlation calculation
    numeric_df = df.select_dtypes(include = np.number)
    plt.figure(figsize = (10, 7))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
    plt.title("Correlation Matrix")
    plt.show()

if boxplot:
    # Categorical vs. Target (Units Sold)
    print("\nAnalyzing relationships between categorical features and Revenue:")
    categorical_features = ["Category", "Location", "Platform"]

    for col in categorical_features:
        plt.figure(figsize=(14, 6))
        # Calculate the order based on median revenue to make the plot more informative
        order = df.groupby(col)["Units Sold"].median().sort_values(ascending=False).index
        sns.boxplot(y=col, x="Units Sold", data=df, order=order, palette="Spectral")
        plt.title(f"Units Sold Distribution by {col}")
        plt.xlabel("Units Sold")
        plt.ylabel(col)
        plt.show()
        print("-" * 130)



# ---------------------------------- Data Preprocessing for ML Model Development ----------------------------------
print_banner("Data Preprocessing for ML Model Development")

# Before feeding the data into the regression models, the data needs to be prepared.  
# 1.  Categorical Encoding:     Convert categorical columns (`Category`, `Location`, `Platform`) into numerical representations. 
#                               We'll use **One-Hot Encoding**, which creates new binary (0 or 1) columns for each unique category 
#                               value.
# 2.  Feature Selection:        Choose the columns (features) we'll use to predict the target (`Units Sold`). We'll drop columns 
#                               that are not useful or redundant (like the original categorical columns after encoding, 'Product 
#                               Name'). We also separate our features (X) from our target variable (y).
# 3.  Train-Test Split:         Divide the dataset into two parts: a **training set** (used to teach the model) and a **testing 
#                               set** (used to evaluate the model's performance on unseen data). This prevents the model from simply 
#                               memorizing the data it was trained on.

# Data Split: Training vs. Testing vs. Validaiton
# 1.  Training:                 Used to train the model.
# 2.  Validation:               Used to tune the model hyperparameters and to evaluate the training process quality as training
#                               proceeds.
# 3.  Testing:                  Used to evaluate the final performance of the trained model.  Testing subset has never been seen by 
#                               the model during training.

# One-Hot-Encoding - Converts values such as "color" into columsn with 1's and 0's for the model to understand.  

# We can't simply convert strings like 'red', 'yellow', or 'blue' into integers like 1, 2, and 3.  The model will assume things like:
# red > yellow > blue


# categorical_cols = ["Location"]
# df_processed = df.copy()
# df_processed = pd.get_dummies(df_processed, columns = categorical_cols, drop_first = True)  # drop_first=True helps avoid multicollinearity

# print(df_processed.head())


# ------------- 1. Categorical Encoding (One-Hot Encoding) -------------
print("\n------------- 1. Categorical Encoding (One-Hot Encoding) -------------")
print("""
      Convert categorical columns (`Category`, `Location`, `Platform`) into numerical representations. 
      We'll use **One-Hot Encoding**, which creates new binary (0 or 1) columns for each unique category 
      value.
      \n
      """)

categorical_cols = ["Category", "Location", "Platform"]

# Make a copy to avoid modifying the original df used in EDA plots if needed later
df_processed = df.copy()
df_processed = pd.get_dummies(df_processed, columns = categorical_cols, drop_first = True)  # drop_first=True helps avoid multicollinearity

# Multicollinearity occurs when two or more independent variables in a regression model are highly correlated. 
# This means that one predictor can be linearly predicted from the others with a high degree of accuracy. 
# It becomes a problem because it undermines the statistical significance of individual predictor variables.

print("Shape before encoding:", df.shape)
print("Shape after encoding:", df_processed.shape)
print("Columns added:", list(set(df_processed.columns) - set(df.columns)))

# Display head of encoded dataframe
print("\nFirst 5 rows of the encoded dataset:")
print(df_processed.head())


# ------------- 2. Feature Selection (Define X and y) -------------
print("\n------------- 2. Feature Selection (Define X and y) -------------")
print("""
      Choose the columns (features) we'll use to predict the target (`Units Sold`). We'll drop columns 
      that are not useful or redundant (like the original categorical columns after encoding, 'Product 
      Name'). We also separate our features (X) from our target variable (y).
      \n
      """)

# Target variable
y = df_processed["Units Sold"]

# Features: Drop the original target, 'Product Name' (too specific),  'Date',  'Units Sold', as they might not be useful for prediction
columns_to_drop = [ "Product Name", "Date", "Units Sold"]

X = df_processed.drop(columns = columns_to_drop)

print("Target variable 'y' shape:", y.shape)
print("Features 'X' shape:", X.shape)
print("\nFeatures being used for modeling:")
print(X.columns.tolist())


# 3. ------------- Train-Test Split -------------
print("\n------------- Train-Test Split -------------")
print("""
      Divide the dataset into two parts: a **training set** (used to teach the model) and a **testing 
      set** (used to evaluate the model's performance on unseen data). This prevents the model from simply 
      memorizing the data it was trained on.
      \n
      """)

# test_size = 0.2 means 20% of the data is for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False) # sklearn library

print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)
print("Training target shape (y_train):", y_train.shape)
print("Testing target shape (y_test):", y_test.shape)



# ---------------------------------- Build Linear Regression ----------------------------------
print_banner("Build Linear Regression")

# Check if training/testing data is available
print("Building and evaluating Linear Regression model...")

# 1. Initialize the Linear Regression model
linear_model = LinearRegression()

# 2. Train the model
print("Training the Linear Regression model...")
linear_model.fit(X_train, y_train)
print("Model training complete.")

# 3. Make predictions on the test set
print("Making predictions on the test set...")
y_pred_linear = linear_model.predict(X_test)

# 4. Evaluate the model
print("\nEvaluating Linear Regression Model:")
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)  # Calculate Root Mean Squared Error
r2_linear = r2_score(y_test, y_pred_linear)

print(f"  Mean Absolute Error (MAE): {mae_linear:.2f}")
print(f"  Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_linear:.2f}")
print(f"  R-squared (R²): {r2_linear:.4f}")


# MAE (Mean Absolute Error):        The average absolute difference between the predicted and actual values. An MAE of 500 means, 
#                                   on average, the prediction was off by 500 "dollars" or "units sold". Lower is better.
# MSE (Mean Squared Error):         Similar to MAE but squares the errors before averaging. This penalizes larger errors more 
#                                   heavily. Lower is better. Units are squared (e.g., dollars or units sold squared), making RMSE 
#                                   often more interpretable.
# RMSE (Root Mean Squared Error):   The square root of MSE. It's in the same units as the target variable (Revenue or units sold in 
#                                   our case), making it easier to understand the typical error magnitude. Lower is better.
# R-squared (R²):                   Represents the proportion of the variance in the target variable that is predictable from the 
#                                   independent variables. It ranges from 0 to 1 (or can be negative for very poor models). An R² 
#                                   of 0.85 means that 85% of the variability in "Revenue" or "units sold" can be explained by our 
#                                   model's features. Higher is generally better (closer to 1).

# Let's visualize predictions Vs. actual
plt.figure(figsize = (8, 8))
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title("Linear Regression: Actual vs. Predicted Units Sold")

plt.grid(True)
# plt.show()



# ---------------------------------- Random Forest Model ----------------------------------
print_banner("Random Forest Model")
# Linear Regression assumes a straight-line relationship. What if the relationship between features 
# and revenue is more complex or non-linear? Let's try a more powerful model: Random Forest Regression.

# Concept: Random Forest is an ensemble method. It builds multiple individual decision trees during 
# training. Each tree is trained on a random subset of the data and considers only a random subset of 
# features at each split point. To make a prediction, the Random Forest averages the predictions from 
# all the individual trees. This approach generally leads to:

#     Higher accuracy:      Can capture complex, non-linear patterns.
#     Robustness:           Less prone to overfitting compared to a single deep decision tree because 
#                           errors from individual trees tend to average out.
#     Feature Importance:   Can provide estimates of how important each feature was in making predictions.

# Steps: (Similar to Linear Regression)

#     Initialize:           Create an instance of the RandomForestRegressor. We can specify parameters 
#                           like n_estimators (number of trees) and random_state (for reproducibility). 
#                           We can also add other parameters like max_depth to control tree complexity.
#     Train:                Fit the model to the training data (X_train, y_train).
#     Predict:              Use the trained model to make predictions on the testing data (X_test).
#     Evaluate:             Compare the predictions (y_pred_rf) against the actual values (y_test) using 
#                           the same regression metrics (MAE, MSE, RMSE, R²).
#     Compare:              How does the Random Forest performance compare to the Linear Regression model?


# ------------- 1. Initialize the Random Forest Regressor model -------------
print("\n------------- 1. Initialize the Random Forest Regressor model -------------.")
# n_estimators: number of trees in the forest
# random_state: ensures reproducibility
# max_depth, min_samples_split: control tree complexity to prevent overfitting
rf_model = RandomForestRegressor(random_state = 42)  # Use out-of-bag samples for validation estimate
print("Random Forest Initialized...")

# Note: Hyperparameters like n_estimators, max_depth etc. can be tuned for better performance
# ------------- 2. Train the model -------------
print("\n------------- 2. Train the model -------------")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# ------------- 3. Make predictions on the test set -------------
print("\n------------- 3. Make predictions on the test set -------------")
y_pred_rf = rf_model.predict(X_test)

# ------------- 4. Evaluate the model -------------
print("\n------------- 4. Evaluate the model -------------")
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"  Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"  Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"  R-squared (R²): {r2_rf:.4f}")

# ------------- 5. Compare with Linear Regression -------------
print("\n------------- 5. Compare with Linear Regression -------------")

print(f"Linear Regression R²: {r2_linear:.4f}")
print(f"Random Forest R²:     {r2_rf:.4f} (Test Set)")

print(f"\nLinear Regression RMSE: {rmse_linear:.2f}")
print(f"Random Forest RMSE:     {rmse_rf:.2f}")

if r2_rf > r2_linear:
    print("\nRandom Forest performed better on Test Set (higher R²).")
elif r2_rf < r2_linear:
    print("\nLinear Regression performed better on Test Set (higher R²).")
else:
    print("\nBoth models performed similarly on Test Set based on R².")

# Visualize predictions vs actual for Random Forest
plt.figure(figsize = (8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title("Random Forest: Actual vs. Predicted Units Sold")
plt.grid(True)
# plt.show()




# ---------------------------------- XG-Boost Model ----------------------------------
print_banner("XG-Boost Model")

# ------------- 1. Instantiate the model -------------
print("\n------------- 1. Instantiate the model -------------")
xgb_model = xgb.XGBRegressor(random_state=42)
print("xb-boost model instantiated...")

# ------------- 2. Train the model -------------
print("\n------------- 2. Train the model -------------")
xgb_model.fit(X_train, y_train)
print("Model training complete.")

# ------------- 3. Make predictions on the test set -------------
print("\n------------- 3. Make predictions on the test set -------------")
y_pred_xgb = xgb_model.predict(X_test)

# ------------- 4. Evaluate the model -------------
print("\n ------------- 4. Evaluate the model -------------")
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f" Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f" Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse_xgb:.2f}")
print(f" R-squared (R²): {r2_xgb:.4f}")

# ------------- 5. Compare all models: Linear Regression vs Random Forest vs XGBoost -------------
print("\n------------- 5. Compare all models: Linear Regression vs Random Forest vs XGBoost -------------")

# Print R-squared values
print(f"Linear Regression R²: {r2_linear:.4f}")
print(f"Random Forest R²:    {r2_rf:.4f}")
print(f"XGBoost R²:           {r2_xgb:.4f}\n")

# Print RMSE values
print(f"Linear Regression RMSE: {rmse_linear:.2f}")
print(f"Random Forest RMSE:     {rmse_rf:.2f}")
print(f"XGBoost RMSE:            {rmse_xgb:.2f}\n")

# Find which model performed best based on R²
r2_scores = {'Linear Regression': r2_linear, 'Random Forest': r2_rf, 'XGBoost': r2_xgb}
best_model_r2 = max(r2_scores, key=r2_scores.get)
print(f"Best model based on R²: {best_model_r2}")

# Find which model performed best based on RMSE (lower is better)
rmse_scores = {'Linear Regression': rmse_linear, 'Random Forest': rmse_rf, 'XGBoost': rmse_xgb}
best_model_rmse = min(rmse_scores, key=rmse_scores.get)
print(f"Best model based on RMSE: {best_model_rmse}")




# ---------------------------------- Feature Importance ----------------------------------
print_banner("Feature Importance")

# Calculate and visualize feature importances from Random Forest

print("Calculating and plotting feature importances from Random Forest...")

# 1. Access feature importances
importances = rf_model.feature_importances_

# 2. Create a pandas Series with feature names
feature_names = X_train.columns
feature_importance_series = pd.Series(importances, index = feature_names)

# 3. Sort and plot
sorted_importances = feature_importance_series.sort_values(ascending = False)

print("\nTop 10 Most Important Features:")
print(sorted_importances.head(10))  # Use display for better formatting

plt.figure(figsize = (12, 8))
# Plotting only the top 20 for clarity if there are many features
num_features_to_plot = min(20, len(sorted_importances))
sns.barplot(
    x = sorted_importances.head(num_features_to_plot),
    y = sorted_importances.head(num_features_to_plot).index,
    palette = "mako",
)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title(f"Top {num_features_to_plot} Feature Importances (Random Forest)")
plt.show()
