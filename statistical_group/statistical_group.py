"""
This is our second attempt along the statistical route, which continues to 
uses TF-IDF and logistic regression for predicting. The main idea in this case
is that since the thousands of ICD9 codes makes it hard for prediction, we
would group similar codes together to narrow down the range of predictions and
thus make predictions easier.

It is recommended to run this file through the bash script, which have
more in depth instructions on the related command arguments

Make sure you have the following MIMIC-III dataset from
PhysioNet:
- NOTEEVENTS.csv.gz, which contains clinical notes
- DIAGNOSES_ICD.csv.gz, which contains diagnosis codes
"""
# given that the file is getting bigger the choice here is to modularize more
# of the code into separate functions

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score
from tqdm.auto import tqdm  # Import tqdm for progress bars
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

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

""" this function modularizes the code that converts clinical text into 
TF-IDF features, which either loads a previously trained vectorizer or 
creates a new one, and transforms all data splits consistently

again, one of the reasons we are using TF-IDF is b/c it gives higher weight
to important medical terms that appear in some specific documents
"""
def create_features(train_data, val_data, test_data, vectorizer_file='tfidf_vectorizer_grouped.pkl', load_vectorizer=True):

    if os.path.exists(vectorizer_file) and load_vectorizer:
        # then we load existing vectorizer to save time, and it also 
        # ensures consistent feature space
        print(f"Loading vectorizer from {vectorizer_file}...")
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # transform all data sets using the existing vectorizer
        print("Transforming training data...")
        X_train = vectorizer.transform(train_data['PROCESSED_TEXT'])
        print("Transforming validation data...")
        X_val = vectorizer.transform(val_data['PROCESSED_TEXT'])
        print("Transforming test data...")
        X_test = vectorizer.transform(test_data['PROCESSED_TEXT'])

    else: # we create a new TF-IDF vectorizer from scratch
        print("Creating new TF-IDF vectorizer...")
        # max_features limits to 10,000 most important terms
        # ngram_range captures both single words and two-word phrases
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        
        # transform all data sets using the new vectorizer
        print("Fitting TF-IDF vectorizer on training data...")
        X_train = vectorizer.fit_transform(train_data['PROCESSED_TEXT'])
        print("Transforming validation data...")
        X_val = vectorizer.transform(val_data['PROCESSED_TEXT'])
        print("Transforming test data...")
        X_test = vectorizer.transform(test_data['PROCESSED_TEXT'])
        
        # again, save the vectorizer for future use
        print(f"Saving vectorizer to {vectorizer_file}...")
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)
    
    print(f"TF-IDF features shape: {X_train.shape}")
    return X_train, X_val, X_test, vectorizer

""" as stated above, medical codes have a "long-tail" distribution, thus,
this function attempts to balance predictions by giving higher weights to
rare classes and lower weights to common classes. So that when predicting,
the model doesn't just predict the most common conditions and ignore the 
rare ones
"""
def calculate_class_weights(y_train):
    print("Calculating class weights to handle imbalanced data...")
    
    # encode string labels into numerical values for us to work with
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    # now we may calculate the class weights, which is inversely proportional
    # to class frequencies
    unique_classes = np.unique(y_encoded)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_encoded
    )
    
    # create a dictionary mapping each diagnosis code to its corresponding
    # weight
    weight_dict = {}
    for i in range(len(label_encoder.classes_)):
        diagnosis = label_encoder.classes_[i]
        weight = class_weights[i]
        weight_dict[diagnosis] = weight

    # then at the end, we print some metrics about class imbalance for insights
    freq_counts = Counter(y_train)
    most_common = freq_counts.most_common(5) # Top 5 frequent classes
    least_common = freq_counts.most_common()[:-6:-1] # Bottom 5 rare classes
    
    print("\nClass imbalance information:")
    print(f"Number of classes: {len(weight_dict)}")
    print(f"Top 5 most common classes:")
    for cls, count in most_common:
        print(f"  {cls}: {count} samples, weight: {weight_dict[cls]:.4f}")
    
    print(f"\nTop 5 least common classes:")
    for cls, count in least_common:
        print(f"  {cls}: {count} samples, weight: {weight_dict[cls]:.4f}")
    
    avg_weight = np.mean(list(weight_dict.values()))
    print(f"\nAverage weight: {avg_weight:.4f}")
    
    return weight_dict

""" again, we are using logistic regression b/c it handles multi-class problems
well and provides interpretable results

this function is to set up a logistic regression model with the corresponding 
settings, train it on our data and at the end return the trained classifier model
"""
def train_model(X_train, y_train, max_iter=100, n_jobs=-1, verbose=1, class_weight=None):
    print("Training logistic regression model on full dataset...")
    
    # check if we're using custom weights or built-in balancing
    if isinstance(class_weight, dict):
        print("Using custom class weights based on class frequencies")
        weight_type = class_weight
    else:
        print(f"Using '{class_weight}' class weighting")
        weight_type = class_weight
    
    # Create the logistic regression model
    classifier = LogisticRegression(
        solver='saga',  # using saga, which is efficient for large datasets
        max_iter=max_iter, 
        class_weight=weight_type, # how to handle class imbalance
        random_state=42, # to make sure this is reproduceable
        n_jobs=n_jobs, # number of CPU cores to use
        verbose=verbose, # whether or not to report progress during training
        C=1.0 # how strongly to enforce rules (1.0 = standard)
    )
    
    # train the model and also time how long it takes
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    return classifier

""" this function makes predictions using our trained model and then compares
the predictions to the actual diagnoses. It would also calculate the performance
metrics we are looking for and on top of it which diagnoses are predicted most 
often
"""
def evaluate_model(classifier, X, y, dataset_name="Test"):
    # also time how long prediction takes, for insights
    start_time = time.time()
    print(f"Making predictions on {dataset_name} set...")
    y_pred = classifier.predict(X)
    pred_time = time.time() - start_time
    
    # calculate and print performance metrics
    accuracy = accuracy_score(y, y_pred) # percentage of exactly correct predictions
    f1 = f1_score(y, y_pred, average='weighted') # balances precision and recall
    hamming = hamming_loss(y, y_pred) # fraction of incorrect predictions
    kappa = cohen_kappa_score(y, y_pred) # agreement beyond chance

    print(f"\n{dataset_name} Metrics:")
    print(f"Prediction time: {pred_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # also further analyze which codes the model predicts the most often
    pred_counts = Counter(y_pred)
    print("\nTop 10 most predicted classes:")
    for cls, count in pred_counts.most_common(10):
        # calculate precision for this class
        correct = 0
        # iterate through each true and predicted label pair
        for true, pred in zip(y, y_pred): 
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
        'cohen_kappa': kappa,
        'prediction_time': pred_time
    }

""" this function is to compare the current model with the best previous
model (if any) using F1 score as the main comparison metric, and then
save the current model if it does perform better
"""
def compare_models(current_metrics, best_metrics_file='best_metrics_grouped.pkl'):
    if os.path.exists(best_metrics_file):
        # it means we do have a previous best model to compare with
        with open(best_metrics_file, 'rb') as f:
            best_metrics = pickle.load(f)
        
        # print comparisons for insights
        print("\nComparing with best model so far:")
        print(f"Current accuracy: {current_metrics['accuracy']:.4f}, Best accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Current F1: {current_metrics['f1_score']:.4f}, Best F1: {best_metrics['f1_score']:.4f}")
        
        # using F1 score, to compare and check if our current model is better
        if current_metrics['f1_score'] > best_metrics['f1_score']:
            print("Current model is better! Saving as new best model.")
            
            # update best metrics
            best_metrics = current_metrics
            with open(best_metrics_file, 'wb') as f:
                pickle.dump(best_metrics, f)
            
            # and also save the model with timestamp for version tracking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_file = f"best_model_grouped_{timestamp}.pkl"
            with open(best_model_file, 'wb') as f:
                pickle.dump(current_metrics['model'], f)
            
            return True
        else:
            print("Current model is not better than previous best.")
            return False
    else:
        # it means there is no previous best model, so this is the best by default
        print("No previous best model found. Saving current as best.")
        with open(best_metrics_file, 'wb') as f:
            pickle.dump(current_metrics, f)
        return True
    
""" and here is the rough idea of the main() function:
1. get command line options
2. load and process data
3. group rare codes
4. split data for training/validation/testing
5. create features
6. train the model
7. evaluate the performance and finally,
8. save the results
"""
def main():
    # to take in command line arguments and handle them accordingly
    parser = argparse.ArgumentParser(description='ICD-9 Code Prediction Model Training with Code Grouping')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations for training')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of data')
    parser.add_argument('--model_file', type=str, default='tfidf_logreg_classifier_grouped.pkl', help='Model file path')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--min_count', type=int, default=5, help='Minimum count to keep a code as-is (smaller values = less grouping)')
    parser.add_argument('--use_custom_weights', action='store_true', help='Use custom class weights instead of "balanced"')
    args = parser.parse_args()
    
    # start timing to see how long the process takes, again for insights
    total_start_time = time.time()
    
    # load the clinical data either from cache or by processing the raw data
    clinical_data = load_or_process_data(force_reprocess=args.reprocess)
    
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
    
    # split data into training (80%), validation (10%), and test (10%) sets
    print("Splitting data into train/validation/test sets...")
    train_data, temp_data = train_test_split(clinical_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # create TF-IDF features for each data set
    X_train, X_val, X_test, vectorizer = create_features(
        train_data, val_data, test_data, load_vectorizer=not args.reprocess)
    
    # get the target labels (diagnosis codes)
    y_train = train_data['GROUPED_ICD9']
    y_val = val_data['GROUPED_ICD9']
    y_test = test_data['GROUPED_ICD9']
    
    # check how many unique diagnosis codes we have in each set, and print for 
    # insights
    unique_train = y_train.nunique()
    unique_val = y_val.nunique()
    unique_test = y_test.nunique()
    print(f"Unique grouped ICD-9 codes in training set: {unique_train}")
    print(f"Unique grouped ICD-9 codes in validation set: {unique_val}")
    print(f"Unique grouped ICD-9 codes in test set: {unique_test}")
    
    # set up class weighting to handle imbalanced data
    class_weight = 'balanced'  # default uses sklearn's built-in balancing
    if args.use_custom_weights:
        # where the alternative is to calculate custom weights based on class
        # frequencies
        class_weight = calculate_class_weights(y_train)
    
    # train the logistic regression model
    classifier = train_model(
        X_train, y_train,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        class_weight=class_weight
    )
    
    # save the model for future use, as again, the training process is time 
    # consuming
    print(f"Saving model to {args.model_file}...")
    with open(args.model_file, 'wb') as f:
        pickle.dump(classifier, f)
    
    # test on validation set
    val_metrics = evaluate_model(classifier, X_val, y_val, "Validation")
    # test on test set
    test_metrics = evaluate_model(classifier, X_test, y_test, "Test")
    
    # add the model to the metrics for tracking, and check if this model
    # is the best one so far
    test_metrics['model'] = classifier
    is_best = compare_models(test_metrics)
    
    # show how long the whole process took
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if is_best:
        print("\nSuccess! This model iteration improved performance.")
    else:
        print("\nTraining completed, but did not improve over previous best model.")
    
    print(f"Model path: {args.model_file}")

if __name__ == "__main__":
    main()
