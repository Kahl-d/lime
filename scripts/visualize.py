import pandas as pd
import matplotlib.pyplot as plt


class FeatureImportanceVisualizer:
    def __init__(self, feature_importance_path):
        """
        Initialize the FeatureImportanceVisualizer class.

        Args:
            feature_importance_path (str): Path to the file containing feature importances.
        """
        self.feature_importance_path = feature_importance_path
        self.feature_importances = self.load_feature_importances()

    def load_feature_importances(self):
        """Load feature importances from a file."""
        return pd.read_csv(self.feature_importance_path)

    def plot_feature_importances(self, save_path=None):
        """
        Plot feature importances.

        Args:
            save_path (str): Path to save the plot image. If None, the plot will just be displayed.
        """
        self.feature_importances.sort_values(by="Importance", ascending=False).plot(
            kind="barh", x="Feature", y="Importance", figsize=(10, 8), legend=False
        )
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to: {save_path}")
        else:
            plt.show()


# Example Usage
if __name__ == "__main__":
    visualizer = FeatureImportanceVisualizer("../data/feature_importances.csv")
    visualizer.plot_feature_importances(save_path="../data/feature_importance_plot.png")