# Automated ICD-9 Code Prediction from Clinical Notes

This project leverages NLP and machine learning to predict ICD-9 medical billing codes from clinical discharge summaries in the MIMIC-III dataset. It includes:
- **Text preprocessing** tailored for clinical notes (de-identification, section tagging, abbreviation expansion).
- **Statistical models** (TF-IDF + Logistic Regression) and **deep learning models** (CNN, BiLSTM).
- **Hierarchical code grouping** to handle rare ICD-9 codes.
- Evaluation metrics (Accuracy, F1, Hamming Loss) for model performance.

## Models
1. **Baseline**: TF-IDF + Logistic Regression (`baseline2.py`).
2. **Statistical with Grouping**: Rare code consolidation (`statistical_group.py`).
3. **Neural Networks**: CNN and LSTM architectures (`nn_models.py`).

## Usage
1. **Data**: Download [MIMIC-III](https://mimic.physionet.org/) and place `NOTEEVENTS.csv.gz` and `DIAGNOSES_ICD.csv.gz` in the root directory.
2. **Run Models**:
   - Baseline: `python3 baseline2.py [--reprocess]`
   - Statistical Grouping: `./run_statistical_grouped.sh [--fast|--thorough]`
   - Neural Networks: `./run_nn_models.sh [--model cnn|lstm]`

## Results
- **Best Model**: CNN achieved **39.45% accuracy** (vs. 1.16% baseline).
- Key techniques: Code grouping, medical text preprocessing, and localized pattern detection (CNNs).

## Poster & Report
- See [`505_project_poster.pdf`](505_project_poster.pdf) and [`CS505_milestone1.pdf`](CS505_milestone1.pdf) for details.

## Dependencies
- Python 3.x, pandas, scikit-learn, TensorFlow/Keras, NLTK.

---

*For questions, contact [tcen17@bu.edu].*
