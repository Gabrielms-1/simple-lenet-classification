import os
import random
import shutil

def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into train, validation, and test sets.
    
    Parameters:
    - dataset_dir (str): Path to the dataset root folder (e.g., "data/alzheimer").
    - train_ratio (float): Proportion of data for the training set.
    - valid_ratio (float): Proportion of data for the validation set.
    - test_ratio (float): Proportion of data for the test set.
    - seed (int): Seed for random shuffling for reproducibility.
    """
    random.seed(seed)
    
    # List all subdirectories (each is a class) and skip folders named 'train', 'valid', 'test'
    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ['train', 'valid', 'test']
    ]
    
    # Ensure the destination directories exist
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    split_counts = {
        'train': {},
        'valid': {},
        'test': {}
    }
    
    # Process each class directory
    for cls in classes:
        if cls == ".DS_Store":
            continue
        class_dir = os.path.join(dataset_dir, cls)
        files = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]
        random.shuffle(files)
    
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        n_test = n_total - n_train - n_valid  # This ensures all files go to one of the splits
    
        split_counts['train'][cls] = n_train
        split_counts['valid'][cls] = n_valid
        split_counts['test'][cls] = n_test
    
        # Split the files based on the calculated numbers
        train_files = files[:n_train]
        valid_files = files[n_train:n_train+n_valid]
        test_files = files[n_train+n_valid:]
    
        # Copy the files to the corresponding folder for each split
        for split, split_files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            dest_class_dir = os.path.join(dataset_dir, split, cls)
            if not os.path.exists(dest_class_dir):
                os.makedirs(dest_class_dir)
            for filename in split_files:
                src_path = os.path.join(class_dir, filename)
                dest_path = os.path.join(dest_class_dir, filename)
                shutil.copy2(src_path, dest_path)
    
        print(f"Class '{cls}': total={n_total}, train={n_train}, valid={n_valid}, test={n_test}")

    print("\nDataset Split Summary:")
    print("| Class               | Train | Valid | Test | Total |")
    print("|-----------------------|-------|-------|------|-------|")

    total_train = sum(split_counts['train'].values())
    total_valid = sum(split_counts['valid'].values())
    total_test = sum(split_counts['test'].values())
    total_total = total_train + total_valid + total_test

    for cls in classes:
        if cls == ".DS_Store":
            continue
        train_count = split_counts['train'].get(cls, 0)
        valid_count = split_counts['valid'].get(cls, 0)
        test_count = split_counts['test'].get(cls, 0)
        class_total = train_count + valid_count + test_count
        print(f"| {cls:<21} | {train_count:<5} | {valid_count:<5} | {test_count:<4} | {class_total:<5} |")

    print("|-----------------------|-------|-------|------|-------|")
    print(f"| Total               | {total_train:<5} | {total_valid:<5} | {total_test:<4} | {total_total:<5} |")

if __name__ == "__main__":
    dataset_dir = "data/alzheimer"
    split_dataset(dataset_dir)
