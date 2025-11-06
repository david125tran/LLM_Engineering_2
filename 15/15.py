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
import warnings  # Added to ignore potential warnings during visualization or modeling

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