import pandas as pd
import random
import os
from itertools import product
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class PerturbationHandler:
    def __init__(self, model_path, data_path, save_dir="../data/"):
        """
        Initialize the PerturbationHandler class.

        Args:
            model_path (str): Path to the fine-tuned model directory.
            data_path (str): Path to the CSV file containing the dataset.
            save_dir (str): Directory to save the perturbed datasets.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.save_dir = save_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        self.data = self.load_data()
        self.maskable_columns = self.data.columns.tolist()

    def load_data(self):
        """Load and preprocess the dataset."""
        data = pd.read_csv(self.data_path)
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

    def generate_masked_descriptions(self, row, max_sentences=100, min_subset_size=2, max_subset_size=5):
        """
        Generate masked text descriptions for a single row by varying subset sizes of masked columns.

        Args:
            row (pd.Series): The row to describe.
            max_sentences (int): Maximum number of masked descriptions to generate.
            min_subset_size (int): Minimum number of columns to mask in each description.
            max_subset_size (int): Maximum number of columns to mask in each description.

        Returns:
            list: List of masked text descriptions.
        """
        descriptions = []
        num_columns = len(self.maskable_columns)

        for _ in range(max_sentences):
            subset_size = random.randint(min_subset_size, min(max_subset_size, num_columns))
            mask_columns = random.sample(self.maskable_columns, subset_size)
            description = (
                f"The individual is {'[MASK]' if 'Gender' in mask_columns else ('male' if row['Gender'] == 1 else 'female')} and is "
                f"{'[MASK]' if 'Age' in mask_columns else row['Age']} years old. "
                # Add more description logic here
            )
            descriptions.append(description)
        return descriptions

    def generate_predicted_sentences(self, masked_sentences, top_k=5):
        """
        Generate predictions for masked sentences using the fine-tuned model.

        Args:
            masked_sentences (list): List of masked sentences.
            top_k (int): Number of top predictions to consider for each `[MASK]`.

        Returns:
            list: List of predicted sentences.
        """
        all_predicted_sentences = []

        for sentence in masked_sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            all_predictions = []
            for token_idx in mask_token_index:
                mask_token_logits = logits[0, token_idx, :]
                top_token_ids = torch.topk(mask_token_logits, top_k, dim=0).indices
                top_tokens = self.tokenizer.convert_ids_to_tokens(top_token_ids)
                all_predictions.append(top_tokens)

            combinations = product(*all_predictions)
            for combination in combinations:
                updated_sentence = sentence
                for prediction in combination:
                    updated_sentence = updated_sentence.replace("[MASK]", prediction, 1)
                all_predicted_sentences.append(updated_sentence)

        return all_predicted_sentences

    def clean_and_standardize_data(self, perturbed_data):
        """
        Clean and standardize perturbed data values.

        Args:
            perturbed_data (pd.DataFrame): DataFrame of perturbed values.

        Returns:
            pd.DataFrame: Cleaned and standardized dataset.
        """
        binary_mappings = {
            "Gender": {"male": 0, "female": 1},
            "Smoking Status": {"do not smoke": 0, "are smokers": 1},
        }
        for column, mapping in binary_mappings.items():
            if column in perturbed_data.columns:
                perturbed_data[column] = perturbed_data[column].replace(mapping)
        return perturbed_data

    def save_perturbed_data(self, df, original_index):
        """
        Save the perturbed dataset.

        Args:
            df (pd.DataFrame): The perturbed dataset.
            original_index (int): Index of the original data point.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        file_name = f"perturbed_data_point_{original_index}.csv"
        df.to_csv(os.path.join(self.save_dir, file_name), index=False)
        print(f"Perturbed dataset saved to: {self.save_dir}/{file_name}")

    def run_perturbation(self, row_index=0):
        """Run the perturbation process for a specific row index."""
        row = self.data.iloc[row_index]
        masked_descriptions = self.generate_masked_descriptions(row)
        predicted_sentences = self.generate_predicted_sentences(masked_descriptions)
        perturbed_df = self.clean_and_standardize_data(pd.DataFrame(predicted_sentences))
        self.save_perturbed_data(perturbed_df, row_index)

# Example Usage
if __name__ == "__main__":
    handler = PerturbationHandler(
        model_path="../fine_tuned_model",
        data_path="../data/data_class.csv"
    )
    handler.run_perturbation(row_index=0)