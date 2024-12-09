import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class FeatureImportanceCalculator:
    def __init__(self, data_path, target_model, output_path="../data/"):
        """
        Initialize the FeatureImportanceCalculator class.

        Args:
            data_path (str): Path to the weighted perturbed dataset.
            target_model: Trained model to predict target values.
            output_path (str): Directory to save the dataset with predictions and other results.
        """
        self.data_path = data_path
        self.target_model = target_model
        self.output_path = output_path
        self.weighted_perturbed_df = self.load_weighted_data()

    def load_weighted_data(self):
        """Load the weighted perturbed dataset."""
        return pd.read_csv(self.data_path)

    def predict_target(self):
        """
        Use the target model to predict target values for the perturbed dataset.

        Returns:
            pd.DataFrame: Updated dataset with predicted values.
        """
        # Drop non-feature columns before prediction
        perturbed_features = self.weighted_perturbed_df.drop(columns=['Weight', 'Physical Activity Status'], errors='ignore')
        self.weighted_perturbed_df['Predicted'] = self.target_model.predict(perturbed_features)
        return self.weighted_perturbed_df

    def save_with_predictions(self, file_name="predicted_weighted_perturbed_data.csv"):
        """
        Save the dataset with predictions to the output path.

        Args:
            file_name (str): Name of the output file.
        """
        file_path = f"{self.output_path}/{file_name}"
        self.weighted_perturbed_df.to_csv(file_path, index=False)
        print(f"Dataset with predictions saved to: {file_path}")

    def calculate_feature_importances(self):
        """
        Train a weighted surrogate linear regression model and calculate feature importances.

        Returns:
            pd.Series: Feature importances from the surrogate model.
        """
        # Prepare features, target, and weights
        X = self.weighted_perturbed_df.drop(columns=['Weight', 'Physical Activity Status', 'Predicted'], errors='ignore')
        y = self.weighted_perturbed_df['Predicted']
        weights = self.weighted_perturbed_df['Weight']

        # Train a weighted linear regression model
        surrogate_model = LinearRegression()
        surrogate_model.fit(X, y, sample_weight=weights)

        # Retrieve feature importances
        feature_importances = pd.Series(surrogate_model.coef_, index=X.columns)
        return feature_importances

    def plot_feature_importances(self, feature_importances):
        """
        Plot feature importances calculated from the surrogate model.

        Args:
            feature_importances (pd.Series): Feature importances to plot.
        """
        plt.figure(figsize=(10, 8))
        feature_importances.sort_values().plot(kind='barh')
        plt.title("Feature Importances from Surrogate Model")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run the entire process: predict, calculate feature importances, and plot.
        """
        # Predict target values
        self.predict_target()

        # Save dataset with predictions
        self.save_with_predictions()

        # Calculate feature importances
        feature_importances = self.calculate_feature_importances()
        print("Feature Importances:")
        print(feature_importances)

        # Plot feature importances
        self.plot_feature_importances(feature_importances)


# Example Usage
if __name__ == "__main__":
    # Replace `target_model` with your trained model
    from sklearn.ensemble import RandomForestRegressor

    # Load or define your target model
    target_model = RandomForestRegressor()
    target_model.fit([[0, 1], [1, 0], [1, 1]], [1, 0, 1])  # Placeholder for actual training

    calculator = FeatureImportanceCalculator(
        data_path="../data/weighted_perturbed_data_point_42.csv",
        target_model=target_model,
        output_path="../data/"
    )
    calculator.run()