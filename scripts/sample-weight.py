import pandas as pd
import numpy as np
import os

class WeightCalculator:
    def __init__(self, data_path, perturbed_file_path, output_dir="../data/"):
        """
        Initialize the WeightCalculator class.

        Args:
            data_path (str): Path to the original dataset file.
            perturbed_file_path (str): Path to the perturbed dataset file.
            output_dir (str): Directory to save the weighted dataset.
        """
        self.data_path = data_path
        self.perturbed_file_path = perturbed_file_path
        self.output_dir = output_dir
        self.original_data = self.load_original_data()
        self.perturbed_data = self.load_perturbed_data()

    def load_original_data(self):
        """Load the original dataset."""
        data = pd.read_csv(self.data_path)
        data = data.drop(columns=['SEQN', 'Unnamed: 0'], errors='ignore')
        data = data.rename(columns={
            'RIAGENDR': 'Gender',
            'RIDAGEYR': 'Age',
            'DMDHHSIZ': 'Household Size',
            'INDFMPIR': 'Income Poverty Ratio',
            'BMXBMI': 'Body Mass Index',
            'DSD010': 'Diet Question One',
            'DSD010AN': 'Diet Question Alternate',
            'SMD415': 'Smoking Status',
            'PAD590': 'Physical Activity One',
            'PAD600': 'Physical Activity Two',
            'HUQ010': 'Health Status',
            'restaurant': 'Restaurant Visits',
            'protein': 'Protein Intake',
            'healthy': 'Healthy Food Intake',
            'unhealthy': 'Unhealthy Food Intake',
            'beverage': 'Beverage Consumption',
            'milk': 'Milk Consumption',
            'MCQ010': 'Medical Condition One',
            'MCQ053': 'Medical Condition Two',
            'MCQ092': 'Medical Condition Three',
            'MCQ140': 'Medical Condition Four',
            'active': 'Physical Activity Status'
        })
        return data

    def load_perturbed_data(self):
        """Load the perturbed dataset."""
        return pd.read_csv(self.perturbed_file_path)

    def calculate_similarity(self, original_point, perturbed_data):
        """
        Calculate similarity weights for each perturbed data point based on the original data point.

        Args:
            original_point (pd.Series): The original data point.
            perturbed_data (pd.DataFrame): The perturbed dataset.

        Returns:
            np.ndarray: Array of weights for each perturbed data point.
        """
        # Ensure numeric types for calculations
        perturbed_data = perturbed_data.apply(pd.to_numeric, errors='coerce')
        original_point = original_point.apply(pd.to_numeric, errors='coerce')

        # Align columns
        perturbed_data = perturbed_data[original_point.index]

        # Calculate Euclidean distances
        distances = np.linalg.norm(perturbed_data.values - original_point.values, axis=1)
        # Convert distances to weights
        weights = np.exp(-distances**2)
        return weights

    def add_weights_to_perturbed_data(self, original_point):
        """
        Add similarity weights to the perturbed dataset.

        Args:
            original_point (pd.Series): The original data point.

        Returns:
            pd.DataFrame: Perturbed dataset with an added 'Weight' column.
        """
        # Compute similarity weights
        weights = self.calculate_similarity(original_point, self.perturbed_data)
        # Normalize weights to sum to 1
        weights /= weights.sum()
        # Add weights as a new column
        self.perturbed_data['Weight'] = weights
        return self.perturbed_data

    def save_weighted_data(self, original_index):
        """
        Save the perturbed dataset with weights to a CSV file.

        Args:
            original_index (int): Index of the original data point.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = f"weighted_perturbed_data_point_{original_index}.csv"
        file_path = os.path.join(self.output_dir, file_name)
        self.perturbed_data.to_csv(file_path, index=False)
        print(f"Weighted dataset saved to: {file_path}")

    def run(self, original_index=0):
        """
        Run the weight calculation process for a specific original data point.

        Args:
            original_index (int): Index of the original data point to use for weight calculation.
        """
        original_point = self.original_data.iloc[original_index]
        weighted_data = self.add_weights_to_perturbed_data(original_point)
        self.save_weighted_data(original_index)
        return weighted_data


# Example Usage
if __name__ == "__main__":
    calculator = WeightCalculator(
        data_path="../data/data_class.csv",
        perturbed_file_path="../data/perturbed_data_point_42.csv",
        output_dir="../data/"
    )
    weighted_data = calculator.run(original_index=0)
    print(weighted_data.head())