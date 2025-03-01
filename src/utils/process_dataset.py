import os
from PIL import Image
import multiprocessing

input_dir = 'data/plant_leaves_disease'
output_dir = 'data/plant_leaves_disease/processed'

os.makedirs(output_dir, exist_ok=True)

def process_image(input_path):
    root = os.path.dirname(input_path)
    file = os.path.basename(input_path)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, file)

        try:
            img = Image.open(input_path)
            img_resized = img.resize((400, 400))
            img_resized.save(output_path)
            print(f"Imagem redimensionada e salva: {output_path}")
        except Exception as e:
            print(f"Erro ao processar {input_path}: {e}")

if __name__ == '__main__':
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))

    pool = multiprocessing.Pool(processes=8)
    pool.map(process_image, image_paths)
    pool.close()
    pool.join()

    print("Processamento de imagens concluído.")
