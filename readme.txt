pip install opencv-python==4.8.0.74

pip install deepface

python save_face.py

python recognize.py


face_recognition_system/
│
├── save_face.py          → Capture & save the reference face (h.jpg)
├── recognize.py          → Recognize the face and greet the user
├── preprocess.py         → Image enhancement functions
│
├── h.jpg                 → Saved/enhanced face sample
└── test.jpg              → Auto-generated test capture

Added logging to `Logs.log` (and console) for easier debugging (utils.py).
Implemented face ROI cropping (largest detected face) and enhancement.
Switched DeepFace model to `ArcFace` for improved accuracy and consistency.

