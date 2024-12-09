import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

data = pd.read_csv('/Users/923673423/lime/data/data_class.csv')

data = data.rename(columns={
    'Unnamed: 0': 'Index',
    'SEQN': 'Sequence Number',
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


# Define a function to generate text descriptions for each row
def generate_text_descriptions(data):
    """
    Generate detailed text descriptions for each row in the dataset.

    Args:
        data (pd.DataFrame): The dataset to describe.

    Returns:
        pd.Series: A series of text descriptions for each row.
    """
    descriptions = []

    for _, row in data.iterrows():
        description = (
        f"The individual is {'male' if row['Gender'] == 1 else 'female'} and is {row['Age']} years old. "
        f"They live in a household with {row['Household Size']} members. Their income-to-poverty ratio is {row['Income Poverty Ratio']:.2f}, "
        f"which is {'below average' if row['Income Poverty Ratio'] < 1 else 'moderate' if 1 <= row['Income Poverty Ratio'] <= 3 else 'above average'}. "
        f"Their body mass index (BMI) is {row['Body Mass Index']:.1f}, calculated from their weight and height. This indicates they are "
        f"{'underweight' if row['Body Mass Index'] < 18.5 else 'in the normal range' if 18.5 <= row['Body Mass Index'] < 25 else 'overweight' if 25 <= row['Body Mass Index'] < 30 else 'obese'}. "
        f"They answered '{row['Diet Question One']}' to a question about their dietary habits, and '{row['Diet Question Alternate']}' to an alternate dietary question. "
        f"They currently {'do not smoke' if row['Smoking Status'] == 'No' else 'are smokers'}, and their physical activity includes {row['Physical Activity One']} minutes "
        f"of moderate-intensity activity and {row['Physical Activity Two']} minutes of vigorous-intensity activity weekly. "
        f"Their self-reported health status is {row['Health Status']} out of 5. "
        f"On average, they visit restaurants {row['Restaurant Visits']} times per month and consume {row['Protein Intake']} grams of protein daily. "
        f"Their healthy food intake is {row['Healthy Food Intake']} servings per day, compared to an unhealthy food intake of {row['Unhealthy Food Intake']} servings per day. "
        f"They drink {row['Beverage Consumption']} beverages daily and consume {row['Milk Consumption']} cups of milk daily. "
        f"Their reported medical conditions include: Condition One={row['Medical Condition One']}, Condition Two={row['Medical Condition Two']}, "
        f"Condition Three={row['Medical Condition Three']}, and Condition Four={row['Medical Condition Four']}. "
        f"Their overall physical activity status is {row['Physical Activity Status']}, which reflects their general lifestyle and fitness."
    )
        descriptions.append(description)

    return pd.Series(descriptions)

# Generate text descriptions for the dataset
data['Text_Description'] = generate_text_descriptions(data)

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
from transformers import AutoModelForMaskedLM

# Load BioBERT model
model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Prepare the dataset for training
def preprocess_function(examples):
    return tokenizer(examples["Text_Description"], padding="max_length", truncation=True, max_length=256)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(data[["Text_Description"]])  # Use only the relevant column
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

# Data collator for dynamic masking during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Use evaluation at each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False
)

from transformers.trainer import Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/Users/923673423/lime/fine_tuned_model")
tokenizer.save_pretrained("/Users/923673423/lime/fine_tuned_model")


