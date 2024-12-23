from PIL import Image
import os

def resize_images(folder_path, size=(128, 128)):
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)
                try:
                    with Image.open(image_path) as img:
                        img = img.resize(size, Image.LANCZOS)
                        img.save(image_path)
                        print(f'Resized and saved: {image_path}')
                except Exception as e:
                    print(f'Error processing {image_path}: {e}')

if __name__ == "__main__":
    resize_images(r"C:\Users\sarab\Desktop\uni\facial recog\custom_dataset")
