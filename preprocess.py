import cv2
import numpy as np
import os


def _load_image(input_image):
    """Load image from path or return ndarray."""
    if isinstance(input_image, np.ndarray):
        return input_image.copy()

    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Image not found: {input_image}")
        img = cv2.imread(input_image)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {input_image}")
        return img

    raise TypeError("Input must be a file path or a NumPy ndarray")


def crop_face_from_image(input_image, cascade_path=None):
    """
    Detect the largest face in the image and crop the ROI.
    Returns: cropped face (numpy array)
    """
    img = _load_image(input_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Select largest box
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]

    # Add padding
    pad_w = int(0.15 * w)
    pad_h = int(0.20 * h)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)

    face_roi = img[y1:y2, x1:x2]
    return face_roi


def enhance_image(img):
    """
    Enhance image using grayscale, histogram equalization,
    blur, sharpening, normalization, then convert to 3 channels.
    """

    img = _load_image(img)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Histogram equalization
    gray = cv2.equalizeHist(gray)

    # 3. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(blurred, -1, kernel)

    # 5. Normalize
    norm = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)

    # 6. Convert to 3-channel (ArcFace requirement)
    final = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    return final
