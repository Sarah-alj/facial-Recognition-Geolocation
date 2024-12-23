import face_recognition

# Load an example image
image_path = r'C:\Users\sarab\Desktop\uni\facial recog\custom_dataset\Sarah Aljurbua\Sarah_Aljurbua_01.jpg'
image = face_recognition.load_image_file(image_path)

# Find all face locations
face_locations = face_recognition.face_locations(image)
print(f"Found {len(face_locations)} face(s) in this image.")

