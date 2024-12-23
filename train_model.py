import dlib
import numpy as np
import cv2
import os

# Paths
predictor_path = r'C:\Users\sarab\Desktop\uni\facial recog\shape_predictor_68_face_landmarks.dat'
images_dir = r'C:\Users\sarab\Desktop\uni\facial recog\custom_dataset'
output_model_path = r'C:\Users\sarab\Desktop\uni\facial recog\face_recognition_model.dat'

# Load the shape predictor model and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(r'C:\Users\sarab\Desktop\uni\facial recog\dlib_face_recognition_resnet_model_v1.dat')

# Create a list to hold face descriptors and labels
face_descriptors = []
labels = []

# Load images and extract face descriptors
def extract_face_descriptors(image_path, label):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = np.array(face_recognition_model.compute_face_descriptor(img, shape))
        face_descriptors.append(face_descriptor)
        labels.append(label)

# Process all images in the directory
for person_folder in os.listdir(images_dir):
    person_folder_path = os.path.join(images_dir, person_folder)
    if os.path.isdir(person_folder_path):
        for filename in os.listdir(person_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(person_folder_path, filename)
                extract_face_descriptors(image_path, person_folder)

# Convert lists to numpy arrays
face_descriptors = np.array(face_descriptors)
labels = np.array(labels)

# Save the face descriptors and labels (you will need to load and use these later for training a classifier)
np.save(r'C:\Users\sarab\Desktop\uni\facial recog\face_descriptors.npy', face_descriptors)
np.save(r'C:\Users\sarab\Desktop\uni\facial recog\labels.npy', labels)

print('Face descriptors extraction complete.')
print(f'Face descriptors saved to {r"C:\Users\sarab\Desktop\uni\facial recog\face_descriptors.npy"}')
print(f'Labels saved to {r"C:\Users\sarab\Desktop\uni\facial recog\labels.npy"}')
