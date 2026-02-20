import os
import sys
import logging
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace

from preprocess import enhance_image, crop_face_from_image
from utils import setup_logger, get_project_root


# CONFIGURATION

ROOT = get_project_root()
DATASET_DIR = os.path.join(ROOT, "dataset")
LOG_PATH = os.path.join(ROOT, "Logs.log")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DEEPFACE_MODEL = "ArcFace"                     # best model for accuracy
THRESHOLD = 0.42                               # ArcFace recommended threshold
VERIFY_PARAMS = {
    "model_name": DEEPFACE_MODEL,
    "detector_backend": "skip"                # We handle detection ourselves
}

logger = setup_logger("facer", LOG_PATH)



# CAMERA CAPTURE

def capture_test_image(save_name="test.jpg"):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logger.error("Unable to open webcam.")
        raise RuntimeError("Unable to open webcam.")

    window_name = "Press SPACE to capture | ESC to exit"
    cv2.namedWindow(window_name)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_name)
    logger.info("Camera started. Waiting for SPACE to capture image.")

    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error("Failed to read from camera.")
            break

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            logger.info("ESC pressed — exiting.")
            cam.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        elif k % 256 == 32:
            cv2.imwrite(save_path, frame)
            logger.info(f"Image captured: {save_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return save_path




# FACE RECOGNITION

def recognize_face():
    logger.info("===== RECOGNITION START =====")

    test_image_path = capture_test_image()

    # STEP 1 — CROP FACE FROM TEST IMAGE
    try:
        face_img = crop_face_from_image(test_image_path, cascade_path=CASCADE_PATH)
    except ValueError:
        print("\n❌ No face detected in the captured image. Try again.")
        logger.warning("No face detected in captured image.")
        return

    # STEP 2 — ENHANCE TEST FACE
    enhanced_face = enhance_image(face_img)

    # STEP 3 — CHECK DATASET
    if not os.path.isdir(DATASET_DIR):
        print("Dataset folder missing. Create a folder named 'dataset'.")
        logger.error("Dataset folder missing.")
        return

    results_list = []
    print("\nComparing with dataset images...\n")

    # STEP 4 — COMPARE AGAINST ALL DATASET IMAGES
    for file in os.listdir(DATASET_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        print(f"🔍 Checking: {file}")
        db_img_path = os.path.join(DATASET_DIR, file)

        try:
            # Crop + enhance dataset face
            db_face = crop_face_from_image(db_img_path, cascade_path=CASCADE_PATH)
            db_enh = enhance_image(db_face)

            # DeepFace verification
            result = DeepFace.verify(enhanced_face, db_enh, **VERIFY_PARAMS)
            distance = result.get("distance", 999)
            name = os.path.splitext(file)[0]

            results_list.append((name, distance))
            logger.info(f"Compared with {file}: distance={distance}")

        except Exception as e:
            logger.exception(f"Error processing {file}: {e}")

    # STEP 5 — NO VALID IMAGES
    if len(results_list) == 0:
        print("⚠ No valid dataset images found.")
        logger.warning("No valid dataset images.")
        return

    # STEP 6 — SORT BY BEST (LOWEST) DISTANCE
    results_list.sort(key=lambda x: x[1])
    best_name, best_distance = results_list[0]

    # DISPLAY RESULTS
    print("\n==============================")
    print("       MATCHING RESULTS")
    print("==============================")
    for name, dist in results_list:
        print(f"{name}: distance={dist}")
    print("==============================\n")

    # STEP 7 — DECIDE IF MATCHED
    if best_distance < THRESHOLD:
        print(f"🎉 BEST MATCH: {best_name.upper()}  (distance={best_distance})")
        print("=======================================")
        print(f"✅ WELCOME {best_name.upper()}!")
        print("=======================================\n")
        logger.info(f"Face recognized: {best_name} (distance={best_distance})")
    else:
        print(f"❌ No match below threshold ({THRESHOLD})")
        print("=======================================")
        print("❌ FACE NOT RECOGNIZED")
        print("=======================================\n")
        logger.warning("Face not recognized.")


        logger.info("===== RECOGNITION END =====\n")

# MAIN

if __name__ == "__main__":
    recognize_face()
