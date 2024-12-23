import cv2
import dlib
import numpy as np
import pickle
import requests
import folium

# Load the pre-trained model and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load face encodings and labels
with open('face_encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    face_encodings = data['encodings']
    face_labels = data['labels']

def get_geolocation():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc').split(',')
        latitude = float(location[0])
        longitude = float(location[1])
        return latitude, longitude
    except Exception as e:
        print(f"Failed to get geolocation: {e}")
        return None, None

def show_location_on_map(lat, lon):
    map_obj = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], popup="Detected Location").add_to(map_obj)
    map_file = "geolocation_map.html"
    map_obj.save(map_file)
    print(f"Map saved as {map_file}. Open it in your browser to view the location.")

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)
        face_descriptor_np = np.array(face_descriptor)

        if len(face_encodings) == 0:
            print("No known face encodings available.")
            return None

        distances = np.linalg.norm(np.array(face_encodings) - face_descriptor_np, axis=1)
        min_distance_idx = np.argmin(distances)

        if distances[min_distance_idx] < 0.6:  # Adjust threshold as needed
            label = face_labels[min_distance_idx]
            return label
    return None

# Initialize variables
last_recognized_label = None

cap = cv2.VideoCapture(1)  # Ensure correct index for your webcam
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    label = recognize_face(frame)

    if label and label != last_recognized_label:
        last_recognized_label = label
        print(f"Face recognized as: {label}")
        
        latitude, longitude = get_geolocation()
        if latitude and longitude:
            print(f"Geolocation: Latitude: {latitude}, Longitude: {longitude}")
            show_location_on_map(latitude, longitude)
    
    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
