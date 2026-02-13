import os
import csv
import numpy as np
import scipy.io
import h5py
import pickle
import glob
from tqdm import tqdm

# ==========================================
# UTILS
# ==========================================

def vector_to_pitch_yaw(v):
    norm = np.linalg.norm(v)
    if norm == 0: return 0, 0
    v = v / norm
    x, y, z = v[0], v[1], v[2]
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))
    yaw = np.arctan2(x, -z) 
    return yaw, pitch

def rad2deg(arr):
    return np.degrees(np.array(arr))

# ==========================================
# DATA LOADERS
# ==========================================

class Gaze360Loader:
    def __init__(self, root, extracted_csv, head_supp_csv):
        self.root = root
        self.extracted_csv = extracted_csv
        self.head_supp_csv = head_supp_csv

    def load_original(self):
        meta_path = os.path.join(self.root, 'metadata.mat')
        if not os.path.exists(meta_path):
            print(f"Gaze360 metadata not found at {meta_path}")
            return np.array([]), np.array([])

        print("Loading Gaze360 metadata.mat...")
        mat = scipy.io.loadmat(meta_path, squeeze_me=True, struct_as_record=False)
        gaze_dirs = mat['gaze_dir'] # (N, 3)
        splits = mat['split'] # 0: train, 1: val, 2: test, 3:unused

        yaws, pitches = [], []
        yaws_used, pitches_used = [], []
        for v, split_val in tqdm(zip(gaze_dirs,splits), total =len(gaze_dirs), desc="Parsing Gaze360 Gaze"):
            v_cam = np.array([-v[0], -v[1], v[2]])
            y, p = vector_to_pitch_yaw(v_cam)
            yaws.append(y)
            pitches.append(p)
            if split_val in range(3): # save only used data
                yaws_used.append(y)
                pitches_used.append(p)                
            
        return np.array(yaws), np.array(pitches), np.array(yaws_used), np.array(pitches_used)

    def load_extracted(self, orig_yaws, orig_pitches):
        print("Loading Gaze360 Extracted CSV...")
        meta_path = os.path.join(self.root, 'metadata.mat')
        mat = scipy.io.loadmat(meta_path, squeeze_me=True, struct_as_record=False)
        recs = mat['recording']
        frames = mat['frame']
        rec_names = mat['recordings']
        splits = mat['split']
        
        lookup = {}
        for i in range(len(recs)):
            r_name = rec_names[recs[i]]
            f_id = frames[i]
            key = f"{r_name}_{f_id}" 
            lookup[key] = i

        extracted_indices = set()
        extracted_gaze_y = []
        extracted_gaze_p = []
        
        with open(self.extracted_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                r_name = row['recording']
                f_id = int(row['frame'])
                key = f"{r_name}_{f_id}"
                
                if key in lookup:
                    idx = lookup[key]
                    extracted_gaze_y.append(orig_yaws[idx])
                    extracted_gaze_p.append(orig_pitches[idx])
                    extracted_indices.add(idx)

        diff_gaze_y = []
        diff_gaze_p = []
        for idx, split_val in enumerate(splits):
            if idx not in extracted_indices and split_val in range(3): # discard "unused" images (split_val=3)
                diff_gaze_y.append(orig_yaws[idx])
                diff_gaze_p.append(orig_pitches[idx])
        
        head_y, head_p = [], []
        if os.path.exists(self.head_supp_csv):
            print("Loading Gaze360 Supplementary Head Pose...")
            with open(self.head_supp_csv, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    head_y.append(float(row['head_yaw_rad']))
                    head_p.append(float(row['head_pitch_rad']))

        return np.array(extracted_gaze_y), np.array(extracted_gaze_p), np.array(head_y), np.array(head_p), np.array(diff_gaze_y), np.array(diff_gaze_p)


class ETHXGazeLoader:
    def __init__(self, root, extracted_csv):
        self.root = root
        self.extracted_csv = extracted_csv

    def load_data(self):
        h5_files = glob.glob(os.path.join(self.root, "*.h5"))
        h5_files.sort()
        
        valid_keys = set()
        print("Loading ETH-XGaze Extracted CSV...")
        with open(self.extracted_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # We now track (subject, frame, camera)
                valid_keys.add((row['subject'], int(row['frame']), int(row['camera'])))

        all_gaze_y, all_gaze_p = [], []
        all_head_y, all_head_p = [], []
        ext_gaze_y, ext_gaze_p = [], []
        ext_head_y, ext_head_p = [], []
        dif_gaze_y, dif_gaze_p = [], []
        dif_head_y, dif_head_p = [], []

        for f_path in tqdm(h5_files, desc="Processing ETH-XGaze H5"):
            subj_id = os.path.basename(f_path).split('.')[0]
            with h5py.File(f_path, 'r') as f:
                face_gaze = f['face_gaze'][:] 
                face_head = f['face_head_pose'][:]
                frame_idx = f['frame_index'][:, 0]
                cam_idx   = f['cam_index'][:, 0]

                for i in range(len(frame_idx)):
                    fid = frame_idx[i]
                    cid = int(cam_idx[i]) # Get camera ID for this entry
                    
                    gp, gy = face_gaze[i]
                    hp, hy = face_head[i]
                    
                    all_gaze_p.append(gp); all_gaze_y.append(gy)
                    all_head_p.append(hp); all_head_y.append(hy)
                    
                    if (subj_id, fid, cid) in valid_keys:
                        ext_gaze_p.append(gp); ext_gaze_y.append(gy)
                        ext_head_p.append(hp); ext_head_y.append(hy)
                    else:
                        dif_head_y.append(hy); dif_head_p.append(hp)
                        dif_gaze_y.append(gy); dif_gaze_p.append(gp)                       

        return (np.array(all_gaze_y), np.array(all_gaze_p), np.array(all_head_y), np.array(all_head_p),
                np.array(ext_gaze_y), np.array(ext_gaze_p), np.array(ext_head_y), np.array(ext_head_p),
                np.array(dif_gaze_y), np.array(dif_gaze_p), np.array(dif_head_y), np.array(dif_head_p))


class GazeGeneLoader:
    def __init__(self, root, extracted_csv):
        self.root = root
        self.extracted_csv = extracted_csv
        self.CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])

    def load_data(self):
        subjects = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)) and d.startswith('subject')]
        subjects.sort()
        
        valid_keys = set()
        print("Loading GazeGene Extracted CSV...")
        with open(self.extracted_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                valid_keys.add(row['image_path'])

        all_gaze_y, all_gaze_p = [], []
        all_head_y, all_head_p = [], []
        ext_gaze_y, ext_gaze_p = [], []
        ext_head_y, ext_head_p = [], []
        dif_gaze_y, dif_gaze_p = [], []
        dif_head_y, dif_head_p = [], []

        for subj in tqdm(subjects, desc="Processing GazeGene"):
            label_dir = os.path.join(self.root, subj, 'labels')
            for cam_id in range(9):
                cam_str = f"camera{cam_id}"
                complex_path = os.path.join(label_dir, f'complex_label_{cam_str}.pkl')
                gaze_path = os.path.join(label_dir, f'gaze_label_{cam_str}.pkl')
                
                if not os.path.exists(complex_path) or not os.path.exists(gaze_path): continue
                    
                with open(complex_path, 'rb') as f: complex_data = pickle.load(f)
                with open(gaze_path, 'rb') as f: gaze_data = pickle.load(f)
                
                num_entries = len(complex_data['img_path'])
                for i in range(num_entries):
                    path = complex_data['img_path'][i]
                    
                    R_blender = gaze_data['head_R_mat'][i] # in CCS
                    #R_cv = R_blender @ self.CV_TO_BLENDER
                    #head_vec = R_cv @ np.array([0, 0, -1])
                    head_vec = R_blender @ np.array([0, 0, 1])
                    hy, hp = vector_to_pitch_yaw(head_vec)
                    
                    try:
                        g_L = gaze_data['visual_axis_L'][i]
                        g_R = gaze_data['visual_axis_R'][i]
                        g_avg = (g_L + g_R) / 2.0
                        gy, gp = vector_to_pitch_yaw(g_avg)
                    except KeyError: continue
                        
                    all_head_y.append(hy); all_head_p.append(hp)
                    all_gaze_y.append(gy); all_gaze_p.append(gp)
                    
                    if path in valid_keys:
                        ext_head_y.append(hy); ext_head_p.append(hp)
                        ext_gaze_y.append(gy); ext_gaze_p.append(gp)
                    else:
                        dif_head_y.append(hy); dif_head_p.append(hp)
                        dif_gaze_y.append(gy); dif_gaze_p.append(gp)                       

        return (np.array(all_gaze_y), np.array(all_gaze_p), np.array(all_head_y), np.array(all_head_p),
                np.array(ext_gaze_y), np.array(ext_gaze_p), np.array(ext_head_y), np.array(ext_head_p),
                np.array(dif_gaze_y), np.array(dif_gaze_p), np.array(dif_head_y), np.array(dif_head_p))

# ==========================================
# MAIN
# ==========================================

def save_data(config, output_file='distributions_data.pkl'):
    data = {}

    # --- 1. GAZE360 ---
    print("\n--- Processing Gaze360 ---")
    loader = Gaze360Loader(config['gaze360_root'], config['gaze360_csv'], config['gaze360_head_csv'])
    o_gy, o_gp, ou_gy, ou_gp = loader.load_original()
    e_gy, e_gp, e_hy, e_hp, d_gy, d_gp = loader.load_extracted(o_gy, o_gp)
    
    data['gaze360'] = {
        'orig_gaze': (rad2deg(ou_gy), rad2deg(ou_gp)), # save only used original gazes
        'orig_head': (np.array([]), np.array([])), # Not available
        'extr_gaze': (rad2deg(e_gy), rad2deg(e_gp)),
        'extr_head': (rad2deg(e_hy), rad2deg(e_hp)),
        'diff_gaze': (rad2deg(d_gy), rad2deg(d_gp)),
        'diff_head': (np.array([]), np.array([])), # Not available
    }
    
    # --- 2. ETH-XGaze ---
    print("\n--- Processing ETH-XGaze ---")
    loader = ETHXGazeLoader(config['xgaze_root'], config['xgaze_csv'])
    res = loader.load_data()
    
    data['xgaze'] = {
        'orig_gaze': (rad2deg(res[0]), rad2deg(res[1])),
        'orig_head': (rad2deg(res[2]), rad2deg(res[3])),
        'extr_gaze': (rad2deg(res[4]), rad2deg(res[5])),
        'extr_head': (rad2deg(res[6]), rad2deg(res[7])),
        'diff_gaze': (rad2deg(res[8]), rad2deg(res[9])),
        'diff_head': (rad2deg(res[10]),rad2deg(res[11]))
    }
    
    # --- 3. GazeGene ---
    print("\n--- Processing GazeGene ---")
    loader = GazeGeneLoader(config['gazegene_root'], config['gazegene_csv'])
    res = loader.load_data()

    data['gazegene'] = {
        'orig_gaze': (rad2deg(res[0]), rad2deg(res[1])),
        'orig_head': (rad2deg(res[2]), rad2deg(res[3])),
        'extr_gaze': (rad2deg(res[4]), rad2deg(res[5])),
        'extr_head': (rad2deg(res[6]), rad2deg(res[7])),
        'diff_gaze': (rad2deg(res[8]), rad2deg(res[9])),
        'diff_head': (rad2deg(res[10]),rad2deg(res[11]))
    }

    # Save
    print(f"\nSaving data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print("Done.")

if __name__ == "__main__":
    # === CONFIGURATION ===
    CONFIG = {
        'gaze360_root': './gaze360Dataset',
        'gaze360_csv': './datasets/Gaze360/gaze360_normalized_ALL_USED.csv',
        'gaze360_head_csv': './gaze360Dataset/Processed data/gaze360_supplementary_headpose_ALL_USED.csv',
        'xgaze_root': './xgaze_448_link/train', 
        'xgaze_csv': './datasets/XGaze_448/xgaze_normalized_det_conf_0_8_ALL.csv',
        'gazegene_root': './GazeGeneDataset/GazeGene_FaceCrops',
        'gazegene_csv': './datasets/GazeGene/gazegene_normalized_det_conf_0_8_ALL.csv'
    }
    
    save_data(CONFIG)
