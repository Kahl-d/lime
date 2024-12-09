# **Enhanced LIME: Context-Aware Interpretability for Machine Learning Models**

---

## **Overview**

This repository focuses on enhancing the Local Interpretable Model-agnostic Explanations (LIME) framework to improve its reliability and domain applicability. The project leverages domain-adaptive fine-tuned models, context-aware perturbation techniques, and weighted surrogate models for feature importance analysis. These advancements aim to provide more realistic, interpretable, and stable explanations for machine learning models, especially in domain-specific applications.

---

## **Technical Problem**

The original LIME framework provides local model interpretability but has significant limitations:

1. **Random Perturbations**:
   - Generated inputs often lack realism and violate domain-specific data distributions.

2. **Lack of Context**:
   - Explanations fail to incorporate nuanced domain-specific relationships between features.

3. **Reduced Stability**:
   - Outputs vary inconsistently with minor changes in input data.

---

## **Proposed Solution**

This project introduces the following key enhancements:

- **Domain-Specific Fine-Tuning**: Leverages BioBERT to generate context-aware perturbations.
- **Context-Aware Perturbation Mechanism**: Replaces random changes with plausible, context-specific variations.
- **Weighted Surrogate Model**: Incorporates similarity weights into the surrogate model to improve fidelity.
- **Improved Feature Importance**: Extracts more accurate and stable feature contributions to model predictions.

---

## **Project Structure**

```plaintext
project/
│
├── tabular-trainer.py               # Trains a domain-adapted model
├── get-perturbations.py             # Generates context-aware perturbations
├── calculate-weights.py             # Calculates similarity weights for perturbations
├── calculate-feature-importance.py  # Extracts feature importance using surrogate models
├── visualize-feature-importance.py  # Visualizes feature importance
├── main.py                          # Main script to run the entire pipeline
├── README.md                        # Project documentation
├── LICENSE                          # Open-source license
├── data/                            # Data files
│   ├── data_class.csv               # Original dataset
│   ├── perturbed_data_point_0.csv   # Example perturbed dataset
│   ├── weighted_perturbed_data.csv  # Example weighted dataset
│
└── results/                         # Output results
    ├── fine_tuned_model/            # Saved fine-tuned model
    ├── feature_importances.csv      # Extracted feature importances
    ├── feature_importance_plot.png  # Visualized feature importances

```

## **How to Run**

### **Setup**

1. **Clone the Repository**:
   ```bash
   git clone <your-github-url>
   cd <repository-name>

2. **Install Dependencies**:
```
Create a Python environment and install required libraries:
python -m venv env
source env/bin/activate  # For Windows: .\env\Scripts\activate
pip install -r requirements.txt
```


## **Pipeline Execution**
```
Run the entire pipeline:
python main.py

This will:
	1.	Train a domain-adaptive model (tabular-trainer.py).
	2.	Generate context-aware perturbations (get-perturbations.py).
	3.	Calculate similarity weights (calculate-weights.py).
	4.	Extract and visualize feature importances (calculate-feature-importance.py).
```

 ### ***How to Verify the Output***
```
1. Generated Perturbations
	•	File: data/perturbed_data_point_0.csv
	•	Verify: Contains realistic perturbations generated using BioBERT.

2. Weighted Perturbations
	•	File: data/weighted_perturbed_data_point_0.csv
	•	Verify: Includes similarity weights for each perturbed data point.

3. Feature Importance
	•	File: results/feature_importances.csv
	•	Verify: Provides feature importance scores from the surrogate model.

4. Visualized Feature Importance
	•	File: results/feature_importance_plot.png
	•	Verify: Displays a sorted bar plot of feature importances.

```

## ***License***

***This project is licensed under the MIT License. See the LICENSE file for full details.***


### ***Acknowledgments***
	- Hugging Face for providing the BioBERT pre-trained model.
	- Scikit-learn for easy-to-use surrogate modeling tools.
	- San Francisco State University (SFSU) and Professor Song for guidance in the CSC 890 AI Entrepreneurship course.
 

