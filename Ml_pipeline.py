#@ Gaurav Lute - SCMS
#Important Library
import json
from striprtf.striprtf import rtf_to_text

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# Extract JSON from RTF file
def extract_json_from_rtf(rtf_file):
    with open(rtf_file, 'r') as file:
        rtf_content = file.read()
    json_data = rtf_to_text(rtf_content)
    return json.loads(json_data)

# Load JSON configuration
def load_json_config(file_path):
    return extract_json_from_rtf(file_path)

# Parse JSON for feature handling
def handle_features(config, data):
    for feature, details in config["feature_handling"].items():
        if feature in data.columns and details["is_selected"]:
            if details["feature_variable_type"] == "numerical":
                if details["feature_details"]["missing_values"] == "Impute":
                    strategy = ("mean" if details["feature_details"]["impute_with"] == "Average of values" 
                               else "constant")
                    fill_value = details["feature_details"].get("impute_value", 0)
                    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                    data[feature] = imputer.fit_transform(data[[feature]])
    return data

# Feature reduction
def reduce_features(config, X, y):
    method = config["feature_reduction"].get("feature_reduction_method", "None")
    if method == "PCA":
        pca = PCA(n_components=int(config["feature_reduction"].get("num_of_features_to_keep", X.shape[1])))
        return pca.fit_transform(X)
    elif method == "Tree-based":
        model = RandomForestRegressor(n_estimators=int(config["feature_reduction"].get("num_of_trees", 10)),
                                       max_depth=int(config["feature_reduction"].get("depth_of_trees", None)))
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[-int(config["feature_reduction"].get("num_of_features_to_keep", X.shape[1])):]
        return X[:, indices]
    return X

# Model training and evaluation
def train_and_evaluate(config, X_train, X_test, y_train, y_test):
    results = {}
    for algo, details in config["algorithms"].items():
        if details["is_selected"]:
            if algo == "RandomForestRegressor":
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor()) ])
                param_grid = {
                    "model__n_estimators": [details.get("min_trees", 10), details.get("max_trees", 100)],
                    "model__max_depth": [details.get("min_depth", 5), details.get("max_depth", 20)],
                }

                # Hyperparameter
                grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error")
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                predictions = best_model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                results[algo] = {
                    "Best Params": grid_search.best_params_,
                    "MSE": mse
                }
    return results

# Main execution flow
def main():
    # Given file 
    config_file = "algoparams_from_ui.json.rtf" 
    config = load_json_config(config_file)

    # Load dataset
    dataset = config["design_state_data"]["session_info"].get("dataset", "iris_modified.csv")
    data = pd.read_csv(dataset)

    # Separate target and features
    target = config["design_state_data"]["target"].get("target", "target")

    x = data.drop(columns=[target])
    y = data[target]

    # Identify categorical and numerical features
    categorical_features = x.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x.select_dtypes(include=['number']).columns.tolist()

    # Handle missing values and encode categorical data
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # Transform features
    x = preprocess.fit_transform(x)

    # Reduce features
    x = reduce_features(config["design_state_data"], x, y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_and_evaluate(config["design_state_data"], x_train, x_test, y_train, y_test)

    # Log results
    file = open('output.txt', 'w')
    file.write("Model Evaluation\n")
    for model, metrics in results.items():
        file.write(f"Model: {model}\n")
        print(f"Model: {model}")
        for metric, value in metrics.items():
            file.write(f"{metric}: {value}\n")
            print(f"{metric}: {value}")
    file.close()
# Run function
if __name__ == "__main__":
    main()
