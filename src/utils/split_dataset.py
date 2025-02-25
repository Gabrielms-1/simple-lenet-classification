import os
import random
import shutil

def split_dataset(main_folder_path):
    train_ratio = 0.7
    val_ratio = 0.2
    # test_ratio = 0.1  # Implicitly calculated

    for class_folder in os.listdir(main_folder_path):
        class_folder_path = os.path.join(main_folder_path, class_folder)
        
        # Skip if not a directory
        if not os.path.isdir(class_folder_path):
            continue

        images = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        def move_images(images, set_type):
            set_folder = os.path.join(main_folder_path, set_type, class_folder)
            os.makedirs(set_folder, exist_ok=True)
            for image in images:
                src = os.path.join(class_folder_path, image)
                dst = os.path.join(set_folder, image)
                shutil.move(src, dst)
        
        move_images(train_images, 'train')
        move_images(val_images, 'val')
        move_images(test_images, 'test')
        
        # Remove original class folder
        os.rmdir(class_folder_path)

# Example usage
main_data_folder = "data/fashionmnist/fashionmnist_big/test"
split_dataset(main_data_folder)
