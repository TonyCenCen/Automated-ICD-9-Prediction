"""
Moving on to the neural network models. Since we are considering clinical text that 
has both a spatial and temporal aspect, we've decide to try CNN (for local patterns),
and LSTM (for sequences).

It is recommended to run this file through the bash script, which have
more in depth instructions on the related command arguments

Make sure you have the following MIMIC-III dataset from
PhysioNet:
- NOTEEVENTS.csv.gz, which contains clinical notes
- DIAGNOSES_ICD.csv.gz, which contains diagnosis codes
"""

import pandas as pd
import numpy as np
import re
import nltk
import os
import pickle
import time
import argparse
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score
from tqdm.auto import tqdm  # Import tqdm for progress bars
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# For neural networks
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tqdm.pandas() # to visualize with progress bar

# just a safety check to download the necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')      # for tokenization
nltk.download('stopwords')  # for stopword removal
nltk.download('wordnet')    # for lemmatization

""" ICD9 nodes have a hierarchical structure, and this function figures out
the chapter (broad disease category), and 
the category (more specific group). This allows us to group similar 
diagnoses together when codes are rare.
"""
def extract_icd9_hierarchy(code):
    if pd.isna(code):  # to handle possible empty or invalid values
        return {'chapter': 'unknown', 'category': 'unknown'}
    
    code_str = str(code).strip()
    
    # now we can start the hierarchy extraction process by first figuring out
    # which chapter this code belongs to 
    if code_str.startswith('V'):
        chapter = 'V'  # supplementary classification
    elif code_str.startswith('E'):
        chapter = 'E'  # external causes
    else:
        # now the tedious process of trying to map numeric codes to 
        # their chapters
        try:
            code_num = float(code_str.split('.')[0])
            
            # map the code number to the corresponding standard ICD-9 chapters
            if 1 <= code_num < 140:
                chapter = '001-139'  # Infectious and parasitic diseases
            elif 140 <= code_num < 240:
                chapter = '140-239'  # Neoplasms
            elif 240 <= code_num < 280:
                chapter = '240-279'  # Endocrine, nutritional, metabolic, immunity
            elif 280 <= code_num < 290:
                chapter = '280-289'  # Blood and blood-forming organs
            elif 290 <= code_num < 320:
                chapter = '290-319'  # Mental disorders
            elif 320 <= code_num < 390:
                chapter = '320-389'  # Nervous system and sense organs
            elif 390 <= code_num < 460:
                chapter = '390-459'  # Circulatory system
            elif 460 <= code_num < 520:
                chapter = '460-519'  # Respiratory system
            elif 520 <= code_num < 580:
                chapter = '520-579'  # Digestive system
            elif 580 <= code_num < 630:
                chapter = '580-629'  # Genitourinary system
            elif 630 <= code_num < 680:
                chapter = '630-679'  # Pregnancy, childbirth, puerperium
            elif 680 <= code_num < 710:
                chapter = '680-709'  # Skin and subcutaneous tissue
            elif 710 <= code_num < 740:
                chapter = '710-739'  # Musculoskeletal and connective tissue
            elif 740 <= code_num < 760:
                chapter = '740-759'  # Congenital anomalies
            elif 760 <= code_num < 780:
                chapter = '760-779'  # Perinatal conditions
            elif 780 <= code_num < 800:
                chapter = '780-799'  # Symptoms, signs, ill-defined conditions
            elif 800 <= code_num < 1000:
                chapter = '800-999'  # Injury and poisoning
            else:
                chapter = 'other'  # Other or unknown
        except ValueError: # in case that we can't parse the number
            chapter = 'unknown' 

    # now we move on to get the category, which is usually the first 3 digits
    if '.' in code_str:
        category = code_str.split('.')[0]  # which is the part before the decimal
    else: # if there is no decimal, take first 3 chars or the whole code if 
        # the whole code is shorter
        category = code_str[:3] if len(code_str) > 3 else code_str

    # return both hierarchy levels
    return {
        'chapter': chapter,
        'category': category,
    }

""" looking into the distribution of medical codes more, it is the case that
a few codes comes up very often, but a majority of other codes appear rarely,
and thus codes have what is a "long tail" distribution and we would have to 
somehow handle this if we want to make better predictions

this function is to find codes that appear less than min_count times and map
rare codes to their parent category. ex. rare code "428.21" -> "428", which
reduces the number of possible outputs
"""
def group_rare_codes(codes, min_count=5):
    # we first count how often each code appears
    code_counts = codes.value_counts()

    # then we get the counts of each code that appears less than the 
    # minimum required count
    codes_below_min_count = code_counts[code_counts < min_count]

    # extract the corresponding index of those rare codes (which contains 
    # the code values), and then convert these indices to a set
    rare_code_indices = codes_below_min_count.index
    rare_codes = set(rare_code_indices)
    
    def map_code(code):
        # if this code is rare, replace it with its parent category
        if code in rare_codes:
            hierarchy = extract_icd9_hierarchy(code)
            return hierarchy['category']  # use its category (first 3 digits usually)
        # otherwise, keep the original code
        return code
    
    # apply this mapping to all codes
    return codes.apply(map_code)

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

""" build a CNN model for text classification, which
embeds text tokens into dense vectors,
applies 1D convolution to capture local patterns,
uses global max pooling to extract the most important features, and 
includes dense layers for classification
"""
def build_cnn_model(vocab_size, embedding_dim, max_sequence_length, num_classes):
    model = Sequential() 
    
    # this is our embedding layer, in this case, we are converting
    # tokens to dense vectors to capture semantic relationships
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    
    # a convolutional layer with 128 filters, which should be enough to detect 
    # diverse patterns, and a kernel size of 5 to capture patterns for 5-word
    # medical phrases
    model.add(Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    
    # global max pooling to keep only the strongest feature per channel and 
    # reduce dimensionality
    model.add(GlobalMaxPooling1D())
    
    # a dense layer with 256 units, trial and error number that should balance
    # complexity and overfitting risk
    model.add(Dense(256, activation='relu'))
    
    # a random dropout of 20% of neurons during training to prevent overfitting
    model.add(Dropout(0.2))
    
    # then finally an output layer with softmax activation for 
    # multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model with categorical crossentropy loss (for multi-class)
    # and Adam optimizer (adaptive learning rate)
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

""" this function is to build a bidirectional LSTM model for text classification, 
which first embeds tokens into dense vectors,
processes the sequence using bidirectional LSTM to capture context, and of crouse
includes dense layers for classification
"""
def build_lstm_model(vocab_size, embedding_dim, max_sequence_length, num_classes):
    model = Sequential()
    
    # this is our embedding layer, in this case, we are converting
    # tokens to dense vectors to capture semantic relationships
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    
    # using bidirectional LSTM to process the sequence in both directions,
    # and this will allow us to capture both past and future context for each word
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    
    # dense layer with 128 units
    model.add(Dense(128, activation='relu'))
    
    # again dropout during training to prevent overfitting
    model.add(Dropout(0.3))
    
    # then finally an output layer with softmax activation for 
    # multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model with categorical crossentropy loss (for multi-class)
    # and Adam optimizer (adaptive learning rate)
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

""" this function, instead of making predictions inside this function using the 
trained model, we pass make the predictions before the function and pass in y_pred
for more modularity

but the purpose of the function is still to compare the predictions to the actual
diagnoses, calculate the performance metrics we are looking for and on top of it which diagnoses are predicted most often
"""
def evaluate_model(y_true, y_pred, class_mapping=None):
    # calculate and print performance metrics
    accuracy = accuracy_score(y_true, y_pred) # percentage of exactly correct predictions
    f1 = f1_score(y_true, y_pred, average='weighted') # balances precision and recall
    hamming = hamming_loss(y_true, y_pred) # fraction of incorrect predictions
    kappa = cohen_kappa_score(y_true, y_pred) # agreement beyond chance
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # also further analyze which codes the model predicts the most often
    pred_counts = Counter(y_pred)
    print("\nTop 10 most predicted classes:")
    for cls_idx, count in pred_counts.most_common(10):
        # first need converts index back to original class label if mapping exists
        if class_mapping is not None:
            cls = class_mapping[cls_idx]
        else:
            cls = cls_idx
        
        # calculate precision for this class    
        correct = 0
        # iterate through each true and predicted label pair
        for true, pred in zip(y_true, y_pred): 
            # check if both the true and predicted labels match the current class
            if true == pred and pred == cls:
                correct += 1 # if they match, increment our counter

        if count > 0:
            precision = correct / count
        else:
            precision = 0
        print(f"  {cls}: predicted {count} times, precision: {precision:.4f}")
    
    # return all metrics in a dictionary, just in case these numbers are needed
    # somewhere else
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'hamming_loss': hamming,
        'cohen_kappa': kappa
    }

""" we've previously ran into problems where some labels are rare and htat in 
standard train_test_split, some labels in validation/test would not be in 
training

this function is to ensure that all labels in validation/test are also in training
"""
def stratified_group_split(df, target_col, test_size=0.2, random_state=42):
    # we start by getting the unique labels
    labels = df[target_col].unique()
    
    # initialize empty DataFrames for train and test
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    
    # split each label group separately
    for label in labels:
        # get all samples with this label
        label_df = df[df[target_col] == label]
        
        # in the case there is very few samples (less than 5), keep all in training
        if len(label_df) < 5:
            train_df = pd.concat([train_df, label_df])
        else: # else if there are enough for split, then split this label group
            label_train, label_test = train_test_split(
                label_df, test_size=test_size, random_state=random_state
            )
            train_df = pd.concat([train_df, label_train])
            test_df = pd.concat([test_df, label_test])
    
    return train_df, test_df


""" and here is the rough idea of the main() function:
1. get command line options
2. load and process data
3. group rare codes
4. splits data with stratification
5. tokenizes and encodes text and labels
6. builds, trains, and evaluates neural network models
7. saves models and related artifacts and report performance metrics
"""
def main():
    # to take in command line arguments and handle them accordingly
    parser = argparse.ArgumentParser(description='Neural Network ICD-9 Code Prediction Model')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'lstm'], help='Type of neural network model (cnn or lstm)')
    parser.add_argument('--min_count', type=int, default=10, help='Minimum count to keep a code as-is (smaller values = less grouping)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--seq_length', type=int, default=500, help='Maximum sequence length for text')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--save_path', type=str, default='nn_model_weights.h5', help='Path to save model weights')
    args = parser.parse_args()
    
    # start timing to see how long the process takes, again for insights
    total_start_time = time.time()
    
    # load the clinical data either from cache or by processing the raw data
    clinical_data = load_or_process_data()
    
    # group rare ICD-9 codes to reduce the number of classes
    print(f"Grouping rare ICD-9 codes (min_count={args.min_count})...")
    original_codes_count = clinical_data['ICD9_CODE'].nunique()
    clinical_data['GROUPED_ICD9'] = group_rare_codes(clinical_data['ICD9_CODE'], min_count=args.min_count)
    grouped_codes_count = clinical_data['GROUPED_ICD9'].nunique()
    
    # show how much we reduced the code complexity, just to provide insight
    print(f"Original number of unique ICD-9 codes: {original_codes_count}")
    print(f"Number of unique codes after grouping: {grouped_codes_count}")
    codes_removed = original_codes_count - grouped_codes_count
    reduction_percentage = (codes_removed / original_codes_count) * 100
    print(f"Reduced by {codes_removed} codes ({reduction_percentage:.2f}%)")

    # use custom stratified split to ensure all labels in val/test are in train
    
    # split data into training (80%), validation (10%), and test (10%) sets
    print("Performing stratified split to ensure all labels are represented in training...")
    train_data, temp_data = stratified_group_split(clinical_data, 'GROUPED_ICD9', test_size=0.2, random_state=42)
    val_data, test_data = stratified_group_split(temp_data, 'GROUPED_ICD9', test_size=0.5, random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")

    # a safety check: to verify all validation and test labels exist in training
    train_labels = set(train_data['GROUPED_ICD9'].unique())
    val_labels = set(val_data['GROUPED_ICD9'].unique())
    test_labels = set(test_data['GROUPED_ICD9'].unique())
    
    val_missing = val_labels - train_labels
    test_missing = test_labels - train_labels
    
    # if there are missings, then we do some manual movings of labels across sets
    if val_missing:
        print(f"Warning: Found {len(val_missing)} labels in validation not in training.")
        # move some samples with these labels to training
        for label in val_missing:
            # find samples with this label in validation
            label_samples = val_data[val_data['GROUPED_ICD9'] == label]

            # take the first row from label_samples DataFrame
            first_label_sample = label_samples.iloc[:1]
            # concatenate this sample with the original training data, and
            # this would allow us to move at least one to training
            train_data = pd.concat(
                [train_data,   # original training data
                first_label_sample]  # new sample to add
            )

            # and remmeber to remove from validation
            val_data = val_data[val_data['GROUPED_ICD9'] != label]
    
    # and do the same thing for testing set if there are missing labels
    if test_missing:
        print(f"Warning: Found {len(test_missing)} labels in test not in training.")
        # move some samples with these labels to training
        for label in test_missing:
            # find samples with this label in test
            label_samples = test_data[test_data['GROUPED_ICD9'] == label]

            # take the first row from label_samples DataFrame
            first_label_sample = label_samples.iloc[:1]
            # concatenate this sample with the original training data, and
            # this would allow us to move at least one to training
            train_data = pd.concat(
                [train_data,   # original training data
                first_label_sample]  # new sample to add
            )
            
            # Remove from test
            test_data = test_data[test_data['GROUPED_ICD9'] != label]
    
    # check and print label coverage again for insights
    train_labels = set(train_data['GROUPED_ICD9'].unique())
    val_labels = set(val_data['GROUPED_ICD9'].unique())
    test_labels = set(test_data['GROUPED_ICD9'].unique())
    
    print(f"Final training label count: {len(train_labels)}")
    print(f"Final validation label count: {len(val_labels)}")
    print(f"Final test label count: {len(test_labels)}")
    print(f"All validation labels in training: {val_labels.issubset(train_labels)}")
    print(f"All test labels in training: {test_labels.issubset(train_labels)}")
    
    # tokenize and prepare sequences for neural network
    print("Tokenizing and preparing sequences...")
    max_words = 20000  # max vocabulary size
    max_sequence_length = args.seq_length
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data['PROCESSED_TEXT'])
    
    # save the tokenizer for future use, as the process is very time consuming
    print("Saving tokenizer...")
    with open('nn_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(train_data['PROCESSED_TEXT'])
    X_val_seq = tokenizer.texts_to_sequences(val_data['PROCESSED_TEXT'])
    X_test_seq = tokenizer.texts_to_sequences(test_data['PROCESSED_TEXT'])
    
    # pad sequences to ensure uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)
    
    # encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_data['GROUPED_ICD9'])
    y_val_encoded = label_encoder.transform(val_data['GROUPED_ICD9'])
    y_test_encoded = label_encoder.transform(test_data['GROUPED_ICD9'])
    
    # also save the label encoder for future predictions
    print("Saving label encoder...")
    with open('nn_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # create a dict that maps encoded numerical labels back to their orignal
    # string values
    label_mapping = {}
    # label_encoder.classes_ contains the original unique labels in the order 
    # they were encoded, ex.: ['diabetes', 'hypertension', 'asthma'] might 
    # have been encoded as [0, 1, 2]
    # iterate through each encoded index and its corresponding original label
    for idx, original_label in enumerate(label_encoder.classes_):
        label_mapping[idx] = original_label
    
    # get model parameters and print for insight
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    embedding_dim = args.embedding_dim
    num_classes = len(label_encoder.classes_)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
    
    # build the selected neural network model according to user param
    if args.model_type == 'cnn':
        print("Building CNN model...")
        model = build_cnn_model(vocab_size, embedding_dim, max_sequence_length, num_classes)
    else:
        print("Building LSTM model...")
        model = build_lstm_model(vocab_size, embedding_dim, max_sequence_length, num_classes)
    
    # define callbacks for training, which stops training early when the model
    # stops improving, and saves the best model automatically (not just the final
    # one)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(args.save_path, save_best_only=True)
    ]
    
    # train the model
    print(f"Training {args.model_type.upper()} model...")
    start_time = time.time()
    history = model.fit(
        X_train_padded, y_train_encoded,
        validation_data=(X_val_padded, y_val_encoded),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val_padded, y_val_encoded, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # make predictions on validation set
    y_val_pred = model.predict(X_val_padded).argmax(axis=1)
    
    # evaluate with detailed metrics
    print("\nDetailed Validation Metrics:")
    val_metrics = evaluate_model(y_val_encoded, y_val_pred, label_mapping)
    
    # evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test_encoded, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # make predictions on test set
    y_test_pred = model.predict(X_test_padded).argmax(axis=1)
    
    # evaluate with detailed metrics
    print("\nDetailed Test Metrics:")
    test_metrics = evaluate_model(y_test_encoded, y_test_pred, label_mapping)
    
    # save model summary to a file
    model_summary_lines = []
    def capture_summary_line(line): # a simple function to handle each summary line
        model_summary_lines.append(line)
    # and now generate the model summary, sending each line to our capture func
    model.summary(print_fn=capture_summary_line)
    with open('nn_model_summary.txt', 'w') as f:
        f.write('\n'.join(model_summary_lines))
    
    # save the complete model
    print("Saving complete model...")
    model.save(f'nn_{args.model_type}_model.h5')
    
    # report total execution time for insight
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # a final accuracy printout
    print(f"Neural Network Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
