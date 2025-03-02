import os
from pathlib import Path

def count_samples():
    base_dir = Path("data/alzheimer")
    splits = ["train", "valid", "test"]
    counts = {split: {} for split in splits}
    
    # Supported image extensions
    img_exts = (".jpg", ".jpeg", ".png")
    
    for split in splits:
        split_path = base_dir / split
        if not split_path.exists():
            continue
            
        # Iterate through class directories
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # Count image files
                num_images = sum(
                    len([f for f in class_dir.glob(f"*{ext}") if f.is_file()])
                    for ext in img_exts
                )
                counts[split][class_name] = num_images

    # Print results
    for split in splits:
        print(f"\n{split.capitalize()} set:")
        total = 0
        for class_name, count in counts[split].items():
            print(f"{class_name}: {count}")
            total += count
        print(f"Total {split} images: {total}\n")

if __name__ == "__main__":
    count_samples()
