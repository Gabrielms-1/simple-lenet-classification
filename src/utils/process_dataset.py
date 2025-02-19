import csv
import os

def process_dataset(csv_file, root_dir):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        new_csv = []
        for row in reader:
            if row[0] == 'filename':
                continue
            image_path = os.path.join(root_dir, row[0])
            label = row[1:]
            label = label.index(' 1')
            new_csv.append([image_path, label])

        with open('data/skin_cancer.v2i.multiclass/valid/processed_classes.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_csv)


if __name__ == '__main__':
    csv_file = 'data/skin_cancer.v2i.multiclass/valid/_classes.csv'
    root_dir = 'data/skin_cancer.v2i.multiclass/valid/'
    process_dataset(csv_file, root_dir)