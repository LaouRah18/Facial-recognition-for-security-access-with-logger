import cv2
import os
from preprocess import enhance_image

def save_face():
    # Ask user for name
    name = input("Enter your name: ").strip().lower()

    # Create dataset folder if not exists
    os.makedirs("dataset", exist_ok=True)

    # File path to save the face
    file_path = f"Facer/dataset/{name}.jpg"
    cam = cv2.VideoCapture(0)
    print("Press SPACE to capture your face")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            processed = enhance_image(frame)
            cv2.imwrite(file_path, processed)
            print(f"✅ Saved enhanced face as {file_path}")
            break

        elif key == 27:  # ESC key
            print("❌ Cancelled")
            break

    cam.release()
    cv2.destroyAllWindows()

save_face()
