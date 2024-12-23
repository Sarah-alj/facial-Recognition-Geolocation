import dlib
import numpy as np
import cv2
import os
import requests
import folium  # Import folium
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

# Paths to your model and descriptor files
predictor_path = r'C:\Users\sarab\Desktop\uni\facial recog\shape_predictor_68_face_landmarks.dat'
face_recognition_model_path = r'C:\Users\sarab\Desktop\uni\facial recog\dlib_face_recognition_resnet_model_v1.dat'
face_descriptors_path = r'C:\Users\sarab\Desktop\uni\facial recog\face_descriptors.npy'
labels_path = r'C:\Users\sarab\Desktop\uni\facial recog\labels.npy'

# Load the shape predictor and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load saved face descriptors and labels
face_descriptors = np.load(face_descriptors_path)
labels = np.load(labels_path)

def get_face_embedding(image_path):
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from path: {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = np.array(face_recognition_model.compute_face_descriptor(img, shape))
        return face_descriptor
    return None

def get_geolocation():
    try:
        # Use an API like ipinfo.io to get geolocation
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc').split(',')
        latitude = float(location[0])
        longitude = float(location[1])
        return latitude, longitude
    except Exception as e:
        print(f"Failed to get geolocation: {e}")
        return None, None

def recognize_face(image_path):
    embedding = get_face_embedding(image_path)
    if embedding is not None:
        # Use KNN for face recognition
        classifier = KNeighborsClassifier(n_neighbors=1)  # Example classifier
        classifier.fit(face_descriptors, labels)
        label = classifier.predict([embedding])

        # Get geolocation
        latitude, longitude = get_geolocation()
        print(f"Geolocation: Latitude: {latitude}, Longitude: {longitude}")  # Print geolocation
        
        # Display the location on the map using folium
        if latitude and longitude:
            show_location_on_map(latitude, longitude)
        
        return label[0], latitude, longitude
    return None, None, None

def show_location_on_map(lat, lon):
    # Create a map object centered around the latitude and longitude
    map_obj = folium.Map(location=[lat, lon], zoom_start=12)
    
    # Add a marker to the map at the given location
    folium.Marker([lat, lon], popup="Detected Location").add_to(map_obj)
    
    # Save the map to an HTML file
    map_file = "geolocation_map.html"
    map_obj.save(map_file)
    print(f"Map saved as {map_file}. Open it in your browser to view the location.")

# Test the recognition and geolocation
test_image_path = r'C:\test_img.jpg'
label, lat, lon = recognize_face(test_image_path)
print(f"Recognized Label: {label}, Location: {lat}, {lon}")  # Print the recognized label and location
