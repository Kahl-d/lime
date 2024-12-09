from tabular_trainer import ModelTrainer
from get_perturbations import PerturbationHandler
from calculate_weights import WeightCalculator
from calculate_feature_importance import FeatureImportanceCalculator


def main():
    # File paths and configurations
    data_path = "../data/data_class.csv"
    model_path = "../fine_tuned_model"
    perturbed_save_dir = "../data/"
    weighted_save_dir = "../data/"
    feature_importance_save_dir = "../data/"

    # Step 1: Train a domain-adapted model (if not already done)
    print("Training a domain-adapted model...")
    trainer = ModelTrainer(
        data_path=data_path,
        model_name="dmis-lab/biobert-base-cased-v1.1",
        output_dir=model_path
    )
    trainer.run()

    # Step 2: Generate perturbations for a specific data point
    print("Generating perturbations...")
    perturbation_handler = PerturbationHandler(
        model_path=model_path,
        data_path=data_path,
        save_dir=perturbed_save_dir
    )
    perturbation_handler.run_perturbation(row_index=0)  # Example: Use the first data point

    # Step 3: Calculate weights for perturbed data
    print("Calculating similarity weights...")
    weight_calculator = WeightCalculator(
        data_path=data_path,
        perturbed_file_path=f"{perturbed_save_dir}/perturbed_data_point_0.csv",  # Adjust based on row index
        output_dir=weighted_save_dir
    )
    weight_calculator.run(original_index=0)

    # Step 4: Calculate and visualize feature importances
    print("Calculating feature importances...")
    from sklearn.ensemble import RandomForestRegressor  # Example target model
    target_model = RandomForestRegressor()
    target_model.fit([[0, 1], [1, 0], [1, 1]], [1, 0, 1])  # Placeholder for actual training

    feature_calculator = FeatureImportanceCalculator(
        data_path=f"{weighted_save_dir}/weighted_perturbed_data_point_0.csv",
        target_model=target_model,
        output_path=feature_importance_save_dir
    )
    feature_calculator.run()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()