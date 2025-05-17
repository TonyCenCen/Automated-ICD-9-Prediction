"""
This is our baseline, which uses TF-IDF and logistic regression for predicting
ICD9 medical codes from clinical notes

To run this file, make sure you have the following MIMIC-III dataset from
PhysioNet:
- NOTEEVENTS.csv.gz, which contains clinical notes
- DIAGNOSES_ICD.csv.gz, which contains diagnosis codes
"""

import pandas as pd
import numpy as np
import re
import nltk
import os
import argparse
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score
from tqdm.auto import tqdm

tqdm.pandas() # to visualize with progress bar

# just a safety check to download the necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')      # for tokenization
nltk.download('stopwords')  # for stopword removal
nltk.download('wordnet')    # for lemmatization

""" this function is to preprocess the given clinical notes while try to 
preserve as much clinically relevant information as possible
"""
def preprocess_medical_text(text):
    # 1. we remove de-identification patterns - these are in brackets 
    # with ** markers
    # ex. [**Patient Name**] or [**2010-03-25**] would be removed
    deid_pattern = r'\[\*\*.*?\*\*\]'  # Matches any text between [** ... **]
    text = re.sub(deid_pattern, '', text)

    # 2. we remove standard header sections and metadata that don't 
    # contribute to diagnosis
    headers_to_remove = [
        # the following are patient metadata headers
        r'Admission Date:.*?(?=\n|Service:|$)', # Admission date and context
        r'Discharge Date:.*?(?=\n|Service:|$)', # Discharge date and context  
        r'Date of Birth:.*?(?=\n|Service:|$)', # Patient birth date  
        r'Sex:.*?(?=\n|Service:|$)', # Patient sex/gender  

        # and these are administrative headers  
        r'Service:.*?(?=\n|HISTORY|$)', # Hospital service/department  
        r'Dictated By:.*?(?=\n|$)', # Author attribution  
        r'JOB#:.*?(?=\n|$)', # Transcription job ID  
        r'MEDQUIST.*?(?=\n|$)' # Transcription service tag  
    ]
    for pattern in headers_to_remove:
        text = re.sub(pattern, '', text)

    # 3. we replace section headers with semantic markers so we can preserve
    # document structure. This should help the model understand which sections
    # of the note are being referred to
    section_headers = [
        # patient history sections, which could be broken down to: 
        # chief complaint & symptoms, and 
        # chronic conditions
        (r'HISTORY OF PRESENT ILLNESS:', ' history_section '),
        (r'PAST MEDICAL HISTORY:', ' past_medical_section '), 
        
        # diagnostic sections, which roughly comprised of:
        # image reports, 
        # specific imaging types like CT, and
        # labs like blood tests
        (r'RADIOLOGIC STUDIES:', ' radiology_section '),
        (r'HEAD CT:', ' head_ct_section '),
        (r'ABDOMINAL CT:', ' abdominal_ct_section '),
        (r'LABORATORY DATA:', ' lab_section '),
        
        # treatment sections, and I put the following under this category:
        # current prescriptions,
        # post-hospital drugs, and
        # drug allergies
        (r'MEDICATIONS:', ' medications_section '),
        (r'DISCHARGE MEDICATIONS:', ' discharge_meds_section '),
        (r'ALLERGIES:', ' allergies_section '),
        
        # clinical assessments:
        # exam findings for say, physical exam,
        # treatment summary
        (r'PHYSICAL EXAMINATION:', ' physical_exam_section '),
        (r'HOSPITAL COURSE:', ' hospital_course_section '),
        
        # and outcomes, which comprised of:
        # final diagnosis, and
        # follow-up plans
        (r'DISCHARGE DIAGNOSIS:', ' discharge_diagnosis_section '),
        (r'PLAN:', ' plan_section ')
    ]
    for pattern, replacement in section_headers:
        text = re.sub(pattern, replacement, text)

    # 4. in this case, we decide to standardize by converting the text to 
    # lowercase
    text = text.lower()

    # 5. in reading through the clinical notes and after several rounds of 
    # preprocessing, it was decided that it is needed here to preserve some
    # important medical patterns before going on to more general processing
    # This step is to prevent these important info from being modified
    # during our later on text processing step

    # first, preserve age patterns, as age is an important number in 
    # determining health
    # (e.g., "81-year-old" -> temporary marker -> restored later)
    age_pattern = r'\d+-year-old'  # matches patterns like "25-year-old"
    age_matches = re.findall(age_pattern, text)
    for i, pattern in enumerate(age_matches):
        text = text.replace(pattern, f'age_{i}')  # temporarily replace with marker

    # preserve medication dosages, as the amount of dosage can tell us the
    # severity of a health problem
    # (e.g., "125 mg IV" -> temporary marker -> restored later)
    med_dose_pattern = r'''
        \d+                      # one or more digits (dose amount)
        \s*                      # to take care of optional whitespace
        (?:mg|mcg|g|ml|units)    # dose units 
        \s*                      # another optional whitespace
        (?:iv|po|sc|im|subq)?    # route of administration may or may not be specified
    '''
    med_matches = re.findall(med_dose_pattern, text, re.VERBOSE) # VERBOSE used for readability
    for i, pattern in enumerate(med_matches):
        text = text.replace(pattern, f'dose_{i}') # temporarily replace with marker

    # preserve lab values with units, as again, these can tell us important info
    # as well; (e.g., "7.2 mg/dl" -> temporary marker -> restored later)
    lab_value_pattern = r'''
        \d+                  # the interger/whole number portion of values
        (?:\.\d+)?           # optional decimal portion (e.g., '.5' in '1.5')
        \s*                  # again, optional whitespace (handles "120mmHg" vs "120 mmHg")
        (?:                  # group of possible units as from the clinical notes:
            mmol             # Millimoles per liter
            |mg/dl           # Milligrams per deciliter  
            |g/dl            # Grams per deciliter
            |ml/min          # Milliliters per minute (e.g., GFR)
            |mmHg            # Millimeters of mercury (blood pressure)
            |cm              # Centimeters
            |mm              # Millimeters
            |kg/m2           # Kilograms per square meter (BMI)
        )
    '''
    lab_matches = re.findall(lab_value_pattern, text, re.VERBOSE)
    for i, pattern in enumerate(lab_matches):
        text = text.replace(pattern, f'lab_value_{i}')  # temporarily replace with marker

    # 6. there is also the need to handle common medical abbreviations by
    # expanding them. As clearly, for the model, 'copd' has a greatly different
    # meaning than 'chronic_obstructive_pulmonary_disease'
    medical_abbreviations = {
        'copd': 'chronic_obstructive_pulmonary_disease',
        'ct': 'computed_tomography',
        'cva': 'cerebrovascular_accident',
        'ekg': 'electrocardiogram',
        'iv': 'intravenous',
        'po': 'oral_administration',
        'bp': 'blood_pressure',
        'hr': 'heart_rate',
        'o2': 'oxygen',
        'mi': 'myocardial_infarction',
        'chf': 'congestive_heart_failure',
        'dvt': 'deep_vein_thrombosis',
        'uti': 'urinary_tract_infection',
        'cabg': 'coronary_artery_bypass_graft',
        'dm': 'diabetes_mellitus',
        'htn': 'hypertension',
        'cad': 'coronary_artery_disease',
        'afib': 'atrial_fibrillation',
        'ckd': 'chronic_kidney_disease'
    }

    # take the above dict, using regex and word boundaries to avoid partial matches,
    # to go through our text to expand the corresponding abbreviations
    for abbr, expansion in medical_abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', expansion, text)

    # 7. clean up any remaining non-alphanumeric characters, but we are 
    # preserving hyphens and underscores, as they may be meaningful in medical 
    # terms
    text = re.sub(r'[^a-zA-Z0-9\-_\s]', ' ', text)

    # 8. now we would tokenize the text into individual words so we can move
    # on to our more general steps of processing
    tokens = word_tokenize(text)

    # 9. the next step is to remove general stopwords but also try to preserve
    # medically important terms, b/c "not fever" would mean the opposite of 
    # having fever
    stop_words = set(stopwords.words('english'))
    medical_terms_to_keep = {'no', 'not', 'with', 'without', 'under', 'over', 'above', 'below', 'positive', 'negative', 'normal', 'abnormal', 'acute', 'chronic'}
    custom_stopwords = stop_words - medical_terms_to_keep
    tokens = [token for token in tokens if token not in custom_stopwords]

    # 10. we messed around a bit and found that lemmatization would allow us
    # to preserve the word meanings better than stemming
    lemmatizer = WordNetLemmatizer()

    # however, we need to keep in mind that there are also terms that we would
    # like to not be lemmatized, and if lemmatized, would have way different 
    # meanings. This includes: expanded abbreviations, disease terms, and 
    # section markers
    protected_terms = set(medical_abbreviations.values())
    protected_terms.update([
        # the disease terms
        'chronic_obstructive_pulmonary_disease', 
        'cerebrovascular_accident',
        'myocardial_infarction', 
        'congestive_heart_failure',
        'deep_vein_thrombosis', 
        'urinary_tract_infection',
        'diabetes_mellitus',
        'hypertension',
        'coronary_artery_disease',
        'atrial_fibrillation',
        'chronic_kidney_disease',
        # and the section markers
        'history_section',
        'past_medical_section',
        'radiology_section',
        'head_ct_section',
        'abdominal_ct_section',
        'medications_section',
        'allergies_section',
        'physical_exam_section',
        'lab_section',
        'hospital_course_section',
        'discharge_meds_section',
        'discharge_diagnosis_section',
        'plan_section'
    ])
    # and now we may apply lemmatization, excluding the protected terms
    new_tokens = []
    for token in tokens:
        if token in protected_terms:
            new_tokens.append(token)
        else:
            new_tokens.append(lemmatizer.lemmatize(token))
    tokens = new_tokens

    # 11. now we may restore the preserved patterns in the processed text
    processed_text = ' '.join(tokens)
    
    # restore age patterns with their original form (e.g., "81yearold")
    for i, pattern in enumerate(age_matches):
        age_value = re.sub(r'[^0-9]', '', pattern)  # we extract just the number
        processed_text = processed_text.replace(f'age_{i}', age_value + 'yearold')
    
    # restore medication dosages (e.g., "125mgIV")
    for i, pattern in enumerate(med_matches):
        simplified_dose = re.sub(r'\s+', '', pattern)  # remove spaces
        processed_text = processed_text.replace(f'dose_{i}', simplified_dose)
    
    # restore lab values (e.g., "7.2mg/dl")
    for i, pattern in enumerate(lab_matches):
        simplified_value = re.sub(r'\s+', '', pattern)  # also remove spaces
        processed_text = processed_text.replace(f'lab_value_{i}', simplified_value)
    
    return processed_text

# this function is to use our preprocessing function to preprocess the notes/data
# and get the data ready for model training
def load_or_process_data(preprocessed_file='ppcd2.csv', force_reprocess=False):
    if os.path.exists(preprocessed_file) and not force_reprocess:
        # we load preprocessed data if available 
        print(f"Loading preprocessed data from {preprocessed_file}...")
        clinical_data = pd.read_csv(preprocessed_file)
        print(f"Loaded {len(clinical_data)} preprocessed clinical records.")
    else:
        # else, we process raw data if preprocessed data not found
        print("Preprocessed data not found. Loading raw data files...")
        
        # load the data files with progress indicators for visual
        print("Loading NOTEEVENTS.csv.gz...")
        notes_df = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip')
        print(f"Loaded {len(notes_df)} notes.")

        print("Loading DIAGNOSES_ICD.csv.gz...")
        diagnoses_df = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip')
        print(f"Loaded {len(diagnoses_df)} diagnosis records.")

        # reading through the available clinical notes, we decided to start from
        # the more comprehensive clinical notes and thus are filtering for
        # for discharge summaries only 
        print("Filtering for discharge summaries...")
        discharge_notes = notes_df[notes_df['CATEGORY'] == 'Discharge summary']
        print(f"Found {len(discharge_notes)} discharge summaries.")

        # safety check to filter out error records, if any
        if 'ISERROR' in discharge_notes.columns:
            print("Removing error records...")
            discharge_notes = discharge_notes[discharge_notes['ISERROR'] != 1]
            print(f"After removing errors: {len(discharge_notes)} records.")

        # then we get primary diagnoses, which is indicated by SEQ_NUM = 1
        print("Extracting primary diagnoses (SEQ_NUM = 1)...")
        primary_diagnoses = diagnoses_df[diagnoses_df['SEQ_NUM'] == 1]
        print(f"Found {len(primary_diagnoses)} primary diagnoses.")

        # now we may join notes with diagnoses based on HADM_ID (hospital admission ID)
        print("Joining notes with diagnoses...")
        clinical_data = pd.merge(
            discharge_notes[['HADM_ID', 'TEXT']], 
            primary_diagnoses[['HADM_ID', 'ICD9_CODE']], 
            on='HADM_ID'
        )
        print(f"Total number of samples after joining: {len(clinical_data)}")

        # now we have gotten the data/notes in the format that we want, we can now
        # move on to our preprocessing step
        print("Preprocessing clinical notes...")
        clinical_data['PROCESSED_TEXT'] = clinical_data['TEXT'].progress_apply(preprocess_medical_text)
        
        # this is a very time consuming step, and therefore, we are saving it as
        # a csv and we may avoid re-preprocessing in the future
        print("Saving preprocessed data...")
        clinical_data[['HADM_ID', 'ICD9_CODE', 'PROCESSED_TEXT']].to_csv(preprocessed_file, index=False)
        print("Preprocessed data saved.")
    
    return clinical_data

def main():
    # handle our only command-line option
    parser = argparse.ArgumentParser(description='ICD-9 Code Prediction Baseline Model')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of data')
    args = parser.parse_args()

    # get the clinical data either from cache or by processing raw data
    clinical_data = load_or_process_data(force_reprocess=args.reprocess)

    # we are going with the decision of spliting the data into:
    # training (80%), validation (10%), and test sets (10%)

    print("Splitting data into train/validation/test sets...")
    train_data, temp_data = train_test_split(clinical_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    # for insights
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")

    # now we can create our TF-IDF vectorizer
    print("Creating TF-IDF features...")
    # max_features=10000 limits to the 10,000 most important terms
    # ngram_range=(1, 2) includes both single words and two-word phrases
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    print("Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_data['PROCESSED_TEXT'])

    print("Transforming validation data...")
    X_val = vectorizer.transform(val_data['PROCESSED_TEXT'])

    print("Transforming test data...")
    X_test = vectorizer.transform(test_data['PROCESSED_TEXT'])
    print(f"TF-IDF features shape: {X_train.shape}")

    # train a logistic regression classifier
    print("Training logistic regression model...")
    classifier = LogisticRegression(
        solver='saga', # using saga, which is efficient for large datasets
        max_iter=30,
        class_weight='balanced', # trying to address class imbalance in ICD codes
        random_state=42, # to make sure this is reproduceable
        n_jobs=-1, #to use all available CPU cores
        verbose=1 # verbose is to show progress during training
    )

    # get our training labels and then subsequently train the model
    y_train = train_data['ICD9_CODE']  # this is our primary diagnosis codes
    y_val = val_data['ICD9_CODE']
    y_test = test_data['ICD9_CODE']
    classifier.fit(X_train, y_train)

    # now we have a model that is trained, we can use it to make predictions
    print("Making predictions on validation set...")
    y_pred_val = classifier.predict(X_val)
    print("Making predictions on test set...")
    y_pred_test = classifier.predict(X_test)

    # evaluate on the validation set and print the metrics for insights
    accuracy_val = accuracy_score(y_val, y_pred_val) # percentage of exactly correct predictions
    f1_val = f1_score(y_val, y_pred_val, average='weighted') # balances precision and recall
    hamming_val = hamming_loss(y_val, y_pred_val) # fraction of incorrect predictions
    kappa_val = cohen_kappa_score(y_val, y_pred_val) # agreement beyond chance
    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy_val:.4f}")
    print(f"F1 Score (weighted): {f1_val:.4f}")
    print(f"Hamming Loss: {hamming_val:.4f}")
    print(f"Cohen's Kappa: {kappa_val:.4f}")

    # and now the true test, we evaluate on test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    hamming_test = hamming_loss(y_test, y_pred_test)
    kappa_test = cohen_kappa_score(y_test, y_pred_test)
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"F1 Score (weighted): {f1_test:.4f}")
    print(f"Hamming Loss: {hamming_test:.4f}")
    print(f"Cohen's Kappa: {kappa_test:.4f}")

    # again, training takes a LOT of time, so we save the trained model 
    # for future use
    print("Saving trained model and vectorizer...")
    pickle.dump(classifier, open('tfidf_logreg_classifier.pkl', 'wb'))
    pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
