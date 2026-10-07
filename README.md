# Ayurveda Project - Phase 1

This repository contains the first phase of the Ayurveda health analysis project, focused on two objectives:

1. **Dataset Integration and Cleaning**
2. **AI/ML Model Training and Evaluation**

## Project Structure

```
Ayurveda_Phase1/
├── data/
│   ├── raw/                    # Original raw datasets
│   ├── integrated/             # Combined/standardized dataset
│   └── cleaned/                # Final cleaned dataset
├── src/
│   ├── integrate_data.py       # Objective 1 - Step 1: Integrate datasets
│   ├── clean_data.py           # Objective 1 - Step 2: Clean dataset
│   ├── train_model.py          # Objective 2 - Step 1: Train ML model
│   └── evaluate_model.py       # Objective 2 - Step 2: Evaluate model
├── models/
│   └── trained_model.pkl       # Saved trained model
├── results/
│   ├── accuracy_results.txt    # Accuracy and CV results
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── cleaning_report.json
│   └── evaluation_report.json
├── requirements.txt
└── README.md
```

## Dataset

The available dataset contains Ayurvedic patient attributes:

- **Raw rows:** 129
- **Rows after cleaning:** 106 (23 duplicates removed)
- **Features:** 18 categorical attributes (Body Size, Body Weight, Eyes, Nose, Lips, Teeth, Skin, Hair, Appetite, Digestion, Thirst, Emotions, Mind, Intellect, Speech, Voice, Dreams, Season Preferred)
- **Target:** `Dosha` (Vata / Pitta / Kapha)

## Results

- **Test Accuracy:** 95.45%
- **Precision:** 95.91%
- **Recall:** 95.45%
- **F1-score:** 95.37%
- **Cross-Validation Accuracy:** 97.14% ± 2.33%

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

Objective 1 - Dataset Integration and Cleaning:

```bash
python src/integrate_data.py
python src/clean_data.py
```

Objective 2 - Model Training and Evaluation:

```bash
python src/train_model.py
python src/evaluate_model.py
```

Or run all steps together:

```bash
python src/integrate_data.py
python src/clean_data.py
python src/train_model.py
python src/evaluate_model.py
```

## Outputs

After running the pipeline, the following outputs are generated:

- `data/integrated/integrated_dataset.csv` — standardized dataset
- `data/cleaned/cleaned_dataset.csv` — final cleaned dataset
- `models/trained_model.pkl` — trained Random Forest classifier
- `results/accuracy_results.txt` — accuracy and cross-validation results
- `results/classification_report.txt` — detailed classification report
- `results/confusion_matrix.png` — confusion matrix visualization
- `results/cleaning_report.json` — data cleaning summary
- `results/evaluation_report.json` — model evaluation summary

## Notes

- This is an academic prototype for a final-year B.E. Artificial Intelligence and Data Science project.
- The model is trained only on the available Ayurvedic dataset.
- Results should not be interpreted as medically validated.
