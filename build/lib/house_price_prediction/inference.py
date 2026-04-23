import pickle
import pandas as pd
import numpy as np
import json
import os


class HousingInference:
    """
    Housing Price Prediction Inference Class
    Loads trained models and preprocessors to make predictions on new data
    """

    def __init__(self, model_path=None):
        """
        Initialize the inference engine by loading pre-trained models and preprocessors
        
        Args:
            model_path: Directory path containing the saved model files.
                       If None, uses the models directory inside the package.
        """
        if model_path is None:
            # Get the directory of this file (where inference.py is)
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models")
        
        self.model_path = model_path
        self.models = {}
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
        self._load_models()

    def _load_models(self):
        """Load all saved models and preprocessors from disk"""
        try:
            with open(f"{self.model_path}/linear_regression_model.pkl", "rb") as f:
                self.models["linear"] = pickle.load(f)
            print("✓ Linear Regression model loaded")

            with open(f"{self.model_path}/svm_regression_model.pkl", "rb") as f:
                self.models["svm"] = pickle.load(f)
            print("✓ SVM Regression model loaded")

            with open(f"{self.model_path}/imputer.pkl", "rb") as f:
                self.imputer = pickle.load(f)
            print("✓ Imputer loaded")

            with open(f"{self.model_path}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("✓ Scaler loaded")
            
            with open(f"{self.model_path}/feature_names.json", "r") as f:
                self.feature_names = json.load(f)
            print(f"✓ Feature names loaded: {self.feature_names}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found. Make sure to run Housing_pred.py first: {e}")

    def _prepare_features(self, data):
        """
        Apply the same feature engineering as training:
        - Handle missing values
        - Add engineered features
        - Encode categorical variables
        
        Args:
            data: DataFrame or dict with housing features
            
        Returns:
            Prepared feature array with correct columns
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        else:
            data = data.copy()

        # Select numerical features (excluding ocean_proximity and income_cat)
        numerical_features = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income'
        ]
        
        data_num = data[numerical_features].copy()

        # Apply imputation
        data_num_imputed = self.imputer.transform(data_num)
        data_num = pd.DataFrame(
            data_num_imputed, columns=numerical_features, index=data.index
        )

        # Add engineered features
        data_num["rooms_per_household"] = (
            data_num["total_rooms"] / data_num["households"]
        )
        data_num["bedrooms_per_room"] = (
            data_num["total_bedrooms"] / data_num["total_rooms"]
        )
        data_num["population_per_household"] = (
            data_num["population"] / data_num["households"]
        )

        # Handle categorical feature (ocean_proximity)
        if "ocean_proximity" in data.columns:
            ocean_dummies = pd.get_dummies(
                data[["ocean_proximity"]], drop_first=True
            )
            data_num = data_num.join(ocean_dummies)
        
        # Ensure all expected features are present (based on self.feature_names)
        for feature in self.feature_names:
            if feature not in data_num.columns:
                data_num[feature] = 0
        
        # Select only the features that were used during training, in the same order
        prepared_data = data_num[self.feature_names].copy()
        
        return prepared_data

    def predict(self, data, model_name="linear"):
        """
        Make predictions using specified model
        
        Args:
            data: Input features (DataFrame or dict)
                 Required columns: longitude, latitude, housing_median_age, total_rooms,
                                   total_bedrooms, population, households, median_income,
                                   ocean_proximity (optional)
            model_name: "linear", "svm", or "all" for both models
            
        Returns:
            Prediction value(s) or dict with both predictions if model_name="all"
        """
        # Prepare features
        prepared_data = self._prepare_features(data)

        if model_name == "all":
            # Return predictions from both models
            linear_pred = self.models["linear"].predict(prepared_data.values)
            svm_data_scaled = self.scaler.transform(prepared_data.values)
            svm_pred = self.models["svm"].predict(svm_data_scaled)
            
            return {
                "linear_regression": float(linear_pred[0]),
                "svm_regression": float(svm_pred[0])
            }
        
        elif model_name == "linear":
            prediction = self.models["linear"].predict(prepared_data.values)
            return float(prediction[0])
        
        elif model_name == "svm":
            prepared_data_scaled = self.scaler.transform(prepared_data.values)
            prediction = self.models["svm"].predict(prepared_data_scaled)
            return float(prediction[0])
        
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose 'linear', 'svm', or 'all'")

    def batch_predict(self, data, model_name="linear"):
        """
        Make predictions on multiple samples
        
        Args:
            data: DataFrame with multiple samples
            model_name: "linear", "svm", or "all"
            
        Returns:
            Array of predictions or dict of arrays
        """
        prepared_data = self._prepare_features(data)

        if model_name == "all":
            linear_preds = self.models["linear"].predict(prepared_data.values)
            svm_data_scaled = self.scaler.transform(prepared_data.values)
            svm_preds = self.models["svm"].predict(svm_data_scaled)
            
            return {
                "linear_regression": linear_preds,
                "svm_regression": svm_preds
            }
        
        elif model_name == "linear":
            return self.models["linear"].predict(prepared_data.values)
        
        elif model_name == "svm":
            prepared_data_scaled = self.scaler.transform(prepared_data.values)
            return self.models["svm"].predict(prepared_data_scaled)
        
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose 'linear', 'svm', or 'all'")


# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    inference = HousingInference()

    # Example 1: Single prediction with dict input
    sample_input = {
        'longitude': -121.89,
        'latitude': 37.29,
        'housing_median_age': 30.0,
        'total_rooms': 5099.0,
        'total_bedrooms': 1111.0,
        'population': 1490.0,
        'households': 1133.0,
        'median_income': 7.2574,
        'ocean_proximity': '<1H OCEAN'
    }

    print("\n=== Single Sample Prediction ===")
    print(f"Input: {sample_input}")
    
    linear_pred = inference.predict(sample_input, model_name="linear")
    print(f"Linear Regression Prediction: ${linear_pred:,.2f}")
    
    svm_pred = inference.predict(sample_input, model_name="svm")
    print(f"SVM Regression Prediction: ${svm_pred:,.2f}")
    
    all_preds = inference.predict(sample_input, model_name="all")
    print(f"All Models: {all_preds}")

    # Example 2: Batch predictions
    print("\n=== Batch Predictions ===")
    batch_data = pd.DataFrame([
        {
            'longitude': -121.89, 'latitude': 37.29, 'housing_median_age': 30.0,
            'total_rooms': 5099.0, 'total_bedrooms': 1111.0, 'population': 1490.0,
            'households': 1133.0, 'median_income': 7.2574, 'ocean_proximity': '<1H OCEAN'
        },
        {
            'longitude': -121.93, 'latitude': 37.05, 'housing_median_age': 6.0,
            'total_rooms': 3237.0, 'total_bedrooms': 647.0, 'population': 1213.0,
            'households': 595.0, 'median_income': 8.3252, 'ocean_proximity': '<1H OCEAN'
        }
    ])
    
    batch_preds = inference.batch_predict(batch_data, model_name="linear")
    print(f"Batch Predictions (Linear): {batch_preds}")
