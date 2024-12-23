import cv2

def test_webcam():
    # Open the webcam (try different index if necessary)
    cap = cv2.VideoCapture(1)  # Try different indexes if 0 does not work

    
    if not cap.isOpened():
        print("Error: Unable to access the webcam. Check the connection or try a different index.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            break
        
        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    test_webcam()