import os
import face_recognition
import pickle

def get_face_encoding(image_path):
    try:
        # Load the image file
        image = face_recognition.load_image_file(image_path)
        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"Detected 0 faces in image: {image_path}")
            return None
        
        # Encode faces in the image
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if face_encodings:
            print(f"Detected {len(face_encodings)} faces in image: {image_path}")
            return face_encodings[0]
        else:
            print(f"Failed to encode face in image: {image_path}")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_face_encodings_from_directory(directory_path):
    face_encodings = {}

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                print(f"Processing image: {file_path}")
                encodings = get_face_encoding(file_path)
                if encodings is not None:
                    face_encodings[file_path] = encodings

    return face_encodings

def save_face_encodings(encodings, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(encodings, f)

if __name__ == "__main__":
    # Set the path to your dataset directory here
    dataset_directory = r'C:\Users\sarab\Desktop\uni\facial recog\custom_dataset'
    
    encodings = generate_face_encodings_from_directory(dataset_directory)
    save_face_encodings(encodings, 'face_encodings.pkl')
    print("Face encodings saved to 'face_encodings.pkl'")
