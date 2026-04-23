import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Ensure data is downloaded before loading
fetch_housing_data()
housing = load_housing_data()

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# Split data using stratified shuffle split on income categories
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Separate features and labels
housing_train = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Prepare housing_prepared with feature engineering
housing_num = housing_train.drop(["ocean_proximity", "income_cat"], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy="median")
housing_num_imputed = imputer.fit_transform(housing_num)

# Create dataframe from imputed data
housing_prepared = pd.DataFrame(
    housing_num_imputed, columns=housing_num.columns, index=housing_train.index
)

# Add new features
housing_prepared["rooms_per_household"] = (
    housing_prepared["total_rooms"] / housing_prepared["households"]
)
housing_prepared["bedrooms_per_room"] = (
    housing_prepared["total_bedrooms"] / housing_prepared["total_rooms"]
)
housing_prepared["population_per_household"] = (
    housing_prepared["population"] / housing_prepared["households"]
)

# Encode categorical feature
housing_cat = housing_train[["ocean_proximity"]]
housing_prepared = housing_prepared.join(
    pd.get_dummies(housing_cat, drop_first=True)
)

# Prepare test set similarly
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop(["ocean_proximity", "income_cat"], axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(housing_prepared)
X_test_scaled = scaler.transform(X_test_prepared)

svm_reg = SVR(kernel="linear", C=1000) 
svm_reg.fit(X_train_scaled, housing_labels)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(f"Linear Regression RMSE: {lin_rmse}")

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(f"Linear Regression MAE: {lin_mae}")

housing_predictions = svm_reg.predict(X_train_scaled)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)

print(f"SVM Regression RMSE: {svm_rmse}")

# Save models and preprocessors to disk
os.makedirs("models", exist_ok=True)

with open("models/linear_regression_model.pkl", "wb") as f:
    pickle.dump(lin_reg, f)

with open("models/svm_regression_model.pkl", "wb") as f:
    pickle.dump(svm_reg, f)

with open("models/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names for inference
import json
feature_names = housing_prepared.columns.tolist()
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("Models and preprocessors saved successfully!")
print("Saved files: linear_regression_model.pkl, svm_regression_model.pkl, imputer.pkl, scaler.pkl, feature_names.json")
print(f"Feature names: {feature_names}")