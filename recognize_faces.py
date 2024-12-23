import dlib
import numpy as np
import cv2
import os
import requests
import folium  # Import folium

# Existing face recognition code...

def get_geolocation():
    try:
        # You can use an API like ipinfo.io to get geolocation
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
        # Load saved face descriptors and labels for comparison
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=1)  # Example classifier
        classifier.fit(face_descriptors, labels)
        label = classifier.predict([embedding])

        # Get geolocation
        latitude, longitude = get_geolocation()
        print(f"Geolocation: Latitude: {latitude}, Longitude: {longitude}")
        
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
    
    # Save the map to an HTML file and open it in the browser
    map_file = "geolocation_map.html"
    map_obj.save(map_file)
    print(f"Map saved as {map_file}. Open it in your browser to view the location.")

# Test the recognition and geolocation
test_image_path = r'C:\test_img.jpg'
label, lat, lon = recognize_face(test_image_path)
print(f"Recognized Label: {label}, Location: {lat}, {lon}")
