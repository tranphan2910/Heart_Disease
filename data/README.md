# Data Folder

This folder contains all datasets used in the Heart Disease Prediction project.

## Datasets

### Original Datasets
- `heart_statlog_cleveland_hungary_final.csv` - Original heart disease dataset (1190 samples, 11 features)
- `heart (1).csv` - Alternative heart disease dataset (1027 samples, 13 features)

### Processed Datasets
- `Cardiovascular_processed.csv` - Preprocessed cardiovascular dataset (1000 samples, 13 features)
- `Cardiovascular_Disease_Dataset.csv` - Raw cardiovascular dataset (before preprocessing)

## Dataset Format

### Required Columns
All datasets must contain these 13 features + target:

**Categorical Features:**
- `sex` - Gender (0: Female, 1: Male)
- `cp` - Chest pain type (0-3)
- `fbs` - Fasting blood sugar > 120 mg/dl (0: No, 1: Yes)
- `restecg` - Resting electrocardiographic results (0-2)
- `exang` - Exercise induced angina (0: No, 1: Yes)
- `slope` - Slope of peak exercise ST segment (0-2)
- `ca` - Number of major vessels colored by fluoroscopy (0-4)
- `thal` - Thalassemia (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Unknown)

**Numerical Features:**
- `age` - Age in years
- `trestbps` - Resting blood pressure (mm Hg)
- `chol` - Serum cholesterol (mg/dl)
- `thalach` - Maximum heart rate achieved
- `oldpeak` - ST depression induced by exercise

**Target:**
- `target` - Heart disease (0: No, 1: Yes)

## Preprocessing

Use the preprocessing script for datasets with different column names:

```bash
python test/preprocess_dataset.py "your_dataset.csv" "processed_dataset.csv"
```

The script automatically:
- Removes patient ID columns
- Maps various column name formats to standard names
- Adds missing columns with default values
- Validates all required columns

## Column Name Variants Supported

The preprocessing script recognizes these variants:
- `gender`, `Gender`, `sex` → `sex`
- `chestpain`, `chest_pain`, `cp` → `cp`
- `restingBP`, `resting_bp`, `trestbps` → `trestbps`
- `serumcholestrol`, `cholesterol`, `chol` → `chol`
- And 20+ more variants...

See `test/preprocess_dataset.py` for complete mapping.

## Notes

- All datasets should use `,` as delimiter
- Missing values should be handled before upload
- Target column must be named `target` (0/1)
- For best results, use datasets with 500+ samples
