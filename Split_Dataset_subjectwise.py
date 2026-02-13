import pandas as pd
import os
import numpy as np

# --- USER CONFIGURATION ------------------------------------------------------
# Paths
INPUT_CSV_PATH = 'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_ALL.csv' #'datasets/GazeGene/gazegene_normalized_det_conf_0_8_ALL.csv' #
OUTPUT_DIR = os.path.dirname(INPUT_CSV_PATH) # Saves in the same folder as input
TRAIN_FILENAME = 'xgaze_normalized_det_conf_0_8_TRAIN.csv' # 'gazegene_normalized_det_conf_0_8_TRAIN.csv' #
TEST_FILENAME = 'xgaze_normalized_det_conf_0_8_TEST.csv'   # 'gazegene_normalized_det_conf_0_8_TEST.csv' #
VALID_FILENAME = 'xgaze_normalized_det_conf_0_8_VALID.csv' # 'gazegene_normalized_det_conf_0_8_VALID.csv' #

# --- SPLIT STRATEGY SETTINGS ---
# Mode 1: "LAST_N" -> Takes the last N subjects (alphabetically) for testing (Default)
# Mode 2: "SPECIFIC" -> Uses a specific list of subject IDs for testing
# Mode 3: "RATIO" -> Randomly selects a percentage of subjects for testing
SPLIT_MODE = "RATIO" 

# Settings for each mode:
# Mode 1: Number of subjects to put in Test and Valid
LAST_N_COUNT = 15
# Mode 2: List strings exactly as they appear in CSV
SPECIFIC_VALID_SUBJECTS = ['subject47', 'subject48','subject49', 'subject50','subject51']
SPECIFIC_TEST_SUBJECTS = ['subject52','subject53', 'subject54', 'subject55', 'subject56']
# Mode 3: 0.2 means 20% of subjects go to Test
TEST_RATIO = 0.2                   
RANDOM_SEED = 42 # Only used if Mode is RATIO

# -----------------------------------------------------------------------------

def main():
    print(f"--- Loading Dataset: {INPUT_CSV_PATH} ---")
    
    # 1. Load Data
    # Note: verify sep matches your generator (semicolon based on your provided files)
    try:
        df = pd.read_csv(INPUT_CSV_PATH, sep=';')
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_CSV_PATH}")
        return

    # 2. Extract Unique Subjects
    if 'subject' not in df.columns:
        print("Error: Column 'subject' not found in CSV.")
        return

    all_subjects = sorted(df['subject'].unique())
    total_subjects = len(all_subjects)
    print(f"Total unique subjects found: {total_subjects}")

    # 3. Determine Split

    if SPLIT_MODE == "LAST_N":
        print(f"Mode: LAST_N (taking last {LAST_N_COUNT} subjects)")
        if LAST_N_COUNT >= total_subjects:
            print("Warning: LAST_N count is larger than total subjects. Using all for test.")
            test_subjects = all_subjects
        else:
            valid_subjects = all_subjects[-LAST_N_COUNT:-LAST_N_COUNT//2]
            test_subjects = all_subjects[-LAST_N_COUNT//2:]
    elif SPLIT_MODE == "SPECIFIC":
        print(f"Mode: SPECIFIC (using defined list)")
        valid_subjects = [s for s in SPECIFIC_VALID_SUBJECTS if s in all_subjects]
        test_subjects = [s for s in SPECIFIC_TEST_SUBJECTS if s in all_subjects]
        
        # Check for typos in user config
        missing = (set(SPECIFIC_TEST_SUBJECTS) | set(SPECIFIC_VALID_SUBJECTS)) - set(all_subjects)
        if missing:
            print(f"Warning: The following configured subjects were not found in data: {missing}")
    elif SPLIT_MODE == "RATIO":
        print(f"Mode: RATIO (randomly selecting {TEST_RATIO*100}%)")
        np.random.seed(RANDOM_SEED)
        test_count = int(total_subjects * TEST_RATIO)
        test_subjects = list(np.random.choice(all_subjects, test_count, replace=False))
        valid_subjects = list(np.random.choice(test_subjects, test_count//2, replace=False))
        test_subjects = list(set(test_subjects)-set(valid_subjects))
    else:
        print(f"Error: Unknown SPLIT_MODE '{SPLIT_MODE}'")
        return

    # 4. Create Train List
    test_subjects_set = set(test_subjects) | set(valid_subjects)
    train_subjects = [s for s in all_subjects if s not in test_subjects_set]

    # 5. Filter DataFrame
    print("\n--- Splitting Data ---")
    train_df = df[df['subject'].isin(train_subjects)]
    valid_df = df[df['subject'].isin(valid_subjects)]
    test_df  = df[df['subject'].isin(test_subjects)]
    
    # 6. Sanity Checks
    print(f"Train Subjects: {len(train_subjects)}")
    print(f"Valid Subjects: {len(valid_subjects)}")
    print(f"Test Subjects:  {len(test_subjects)}")
    print(f"Train Samples:  {len(train_df)}")
    print(f"Valid Samples:  {len(valid_df)}")
    print(f"Test Samples:   {len(test_df)}")
    
    # Check for leakage
    intersection1 = set(train_df['subject'].unique()).intersection(set(test_df['subject'].unique()))
    intersection2 = set(train_df['subject'].unique()).intersection(set(valid_df['subject'].unique()))
    intersection3 = set(test_df['subject'].unique()).intersection(set(valid_df['subject'].unique()))
    if intersection1 or intersection2 or intersection3:
        print(f"CRITICAL ERROR: Data leakage detected. Subjects in both sets: {intersection1|intersection2|intersection3}")
        return
    else:
        print("Verification passed: No subject overlap between Train, Test and Valid.")

    # 7. Save Files
    train_path = os.path.join(OUTPUT_DIR, TRAIN_FILENAME)
    valid_path = os.path.join(OUTPUT_DIR, VALID_FILENAME)
    test_path  = os.path.join(OUTPUT_DIR, TEST_FILENAME)

    print(f"\n--- Saving Files ---")
    train_df.to_csv(train_path, index=False, sep=';')
    print(f"Saved Train: {train_path}")

    valid_df.to_csv(valid_path, index=False, sep=';')
    print(f"Saved Valid:  {valid_path}")
    
    test_df.to_csv(test_path, index=False, sep=';')
    print(f"Saved Test:  {test_path}")

if __name__ == "__main__":
    main()
