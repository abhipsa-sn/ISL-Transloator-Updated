import os
import glob
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from src import config
except ImportError:
    try:
        import config
    except ImportError:
        # Fallback config if not found
        class Config:
            RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
            PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
            TEST_SIZE = 0.2
            RANDOM_STATE = 42
        config = Config()

def load_raw_data(data_type='static'):
    """
    Loads all .npy files from data/raw/{class_name}/ directories
    Returns X (features array), y (labels array), class_names list
    """
    raw_path = config.RAW_DATA_PATH
    if not os.path.exists(raw_path):
        warnings.warn(f"Raw data path does not exist: {raw_path}")
        return np.array([]), np.array([]), []

    X = []
    y = []
    class_names = []

    # Get all subdirectories (classes)
    classes = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    classes.sort()

    for class_name in classes:
        class_names.append(class_name)
        class_path = os.path.join(raw_path, class_name)
        
        # Load all .npy files in this class directory
        pattern = os.path.join(class_path, '*.npy')
        files = glob.glob(pattern)
        
        if not files:
            warnings.warn(f"No .npy files found for class '{class_name}' at {class_path}")
            continue
            
        for file in files:
            try:
                data = np.load(file)
                # For static, data should be a single frame of landmarks. 
                # If it has shape (N, features) and data_type is static, we might take each row
                if data_type == 'static':
                    if len(data.shape) > 1:
                        for row in data:
                            X.append(row)
                            y.append(class_name)
                    else:
                        X.append(data)
                        y.append(class_name)
                elif data_type == 'dynamic':
                    X.append(data)
                    y.append(class_name)
            except Exception as e:
                warnings.warn(f"Error loading {file}: {e}")

    return np.array(X), np.array(y), class_names

def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist position (landmark 0) for each hand.
    Scale by max distance from wrist to normalize hand size.
    Assumes landmarks shape is (126,) -> (21*3 for right hand, 21*3 for left hand)
    """
    if len(landmarks.shape) == 1:
        # Assuming 126 features: 63 for right hand, 63 for left hand
        normalized = np.zeros_like(landmarks)
        
        # Process each hand separately (0:63 and 63:126)
        for i in range(0, 126, 63):
            hand_landmarks = landmarks[i:i+63]
            if np.all(hand_landmarks == 0):
                continue
                
            # Wrist is the first landmark (0, 1, 2)
            wrist_x, wrist_y, wrist_z = hand_landmarks[0:3]
            
            # Subtract wrist from all landmarks
            hand_landmarks_shifted = hand_landmarks.copy()
            for j in range(0, 63, 3):
                hand_landmarks_shifted[j] -= wrist_x
                hand_landmarks_shifted[j+1] -= wrist_y
                hand_landmarks_shifted[j+2] -= wrist_z
                
            # Scale by max distance from wrist
            distances = []
            for j in range(0, 63, 3):
                dist = np.sqrt(hand_landmarks_shifted[j]**2 + 
                             hand_landmarks_shifted[j+1]**2 + 
                             hand_landmarks_shifted[j+2]**2)
                distances.append(dist)
            max_dist = max(distances) if distances else 0
            
            if max_dist > 0:
                hand_landmarks_shifted = hand_landmarks_shifted / max_dist
                
            normalized[i:i+63] = hand_landmarks_shifted
            
        return normalized
    elif len(landmarks.shape) == 2:
        return np.array([normalize_landmarks(row) for row in landmarks])
    else:
        return landmarks

def augment_data(X, y, augmentation_factor=3):
    """
    Add Gaussian noise, random scaling, and random translation.
    Returns augmented X, y.
    """
    print(f"Augmenting data with factor {augmentation_factor}...")
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(augmentation_factor - 1):
        # Gaussian noise
        noise = np.random.normal(0, 0.01, X.shape)
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1, X.shape)
        
        # Random translation
        translation = np.random.normal(0, 0.02, X.shape)
        
        X_new = (X * scale) + noise + translation
        X_aug.append(X_new)
        y_aug.append(y)
        
    return np.vstack(X_aug), np.concatenate(y_aug)

def prepare_dataset(data_type='static', augment=True):
    """
    Full pipeline: load -> normalize -> augment -> encode labels -> train/test split
    Saves processed data to data/processed/
    """
    print("Loading raw data...")
    X, y, class_names = load_raw_data(data_type)
    
    if len(X) == 0:
        print("No data loaded. Please check data/raw/ directory.")
        return None, None, None, None, None
        
    print(f"Loaded {len(X)} samples across {len(class_names)} classes.")
    
    print("Normalizing landmarks...")
    X = normalize_landmarks(X)
    
    if augment and data_type == 'static':
        X, y = augment_data(X, y, augmentation_factor=3)
        print(f"After augmentation: {len(X)} samples.")
        
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=getattr(config, 'TEST_SIZE', 0.2), 
        random_state=getattr(config, 'RANDOM_STATE', 42), stratify=y_encoded
    )
    
    # Save processed data
    proc_path = config.PROCESSED_DATA_PATH
    os.makedirs(proc_path, exist_ok=True)
    
    np.save(os.path.join(proc_path, 'X_train.npy'), X_train)
    np.save(os.path.join(proc_path, 'X_test.npy'), X_test)
    np.save(os.path.join(proc_path, 'y_train.npy'), y_train)
    np.save(os.path.join(proc_path, 'y_test.npy'), y_test)
    np.save(os.path.join(proc_path, 'label_encoder.npy'), le.classes_)
    
    print(f"Processed data saved to {proc_path}")
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    print(f"Classes: {le.classes_}")
    
    return X_train, X_test, y_train, y_test, le

if __name__ == '__main__':
    print("Starting Preprocessing Pipeline...")
    prepare_dataset(data_type='static', augment=True)
    print("Preprocessing completed successfully.")
