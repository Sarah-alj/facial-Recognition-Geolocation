import cv2
import os

def load_and_label_data(dataset_path):
    # Initialize lists to store data and labels
    images = []
    labels = []
    label_map = {}

    # Get list of folders in the dataset
    folders = os.listdir(dataset_path)
    for folder_index, folder_name in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label_map[folder_index] = folder_name
            # Load all images in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                # Load image using OpenCV
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(folder_index)
                else:
                    print(f'Error loading image: {image_path}')
    
    return images, labels, label_map

if __name__ == "__main__":
    dataset_path = "C:/Users/sarab/Desktop/uni/facial recog/custom_dataset"
    images, labels, label_map = load_and_label_data(dataset_path)
    print(f'Loaded {len(images)} images from {len(label_map)} labels.')
    print('Label Map:', label_map)
