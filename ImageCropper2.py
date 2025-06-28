import sys
import subprocess

def ensure_dependencies():
    import importlib
    required = ['cv2', 'numpy']
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            pip_pkg = 'opencv-python' if pkg == 'cv2' else pkg
            print(f"Installing missing package: {pip_pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_pkg])

ensure_dependencies()

import cv2
import os

def detect_and_crop_faces(img, padding=20, output_size=(300, 400), x_offset=0, y_offset=0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        # Center crop a fixed-size box around the face, with offsets
        center_x = x + w // 2 + x_offset
        center_y = y + h // 2 + y_offset
        crop_w, crop_h = output_size

        x1 = max(center_x - crop_w // 2, 0)
        y1 = max(center_y - crop_h // 2, 0)
        x2 = min(x1 + crop_w, img.shape[1])
        y2 = min(y1 + crop_h, img.shape[0])

        # Adjust if crop goes out of bounds
        if x2 - x1 < crop_w:
            x1 = max(x2 - crop_w, 0)
        if y2 - y1 < crop_h:
            y1 = max(y2 - crop_h, 0)

        face_img = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face_img, output_size)
        cropped_faces.append(face_resized)
        break  # Only process the first detected face

    return cropped_faces


def batch_process_employee_headshots(root_dir, output_dir, x_offset=0, y_offset=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for employee_folder in os.listdir(root_dir):
        employee_path = os.path.join(root_dir, employee_folder)
        if not os.path.isdir(employee_path):
            continue

        # Normalize employee name (remove spaces or slashes)
        employee_name = employee_folder.replace(' ', '_')

        # Create output subdirectory for this employee
        employee_output_dir = os.path.join(output_dir, employee_name)
        if not os.path.exists(employee_output_dir):
            os.makedirs(employee_output_dir)

        image_counter = 1
        for file_name in os.listdir(employee_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(employee_path, file_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not read {image_path}")
                    continue

                faces = detect_and_crop_faces(img, x_offset=x_offset, y_offset=y_offset)
                for face in faces:
                    output_filename = f"{employee_name}_{image_counter}.jpg"
                    output_path = os.path.join(employee_output_dir, output_filename)
                    cv2.imwrite(output_path, face)
                    print(f"Saved: {output_path}")
                    image_counter += 1

# Example usage with offset
batch_process_employee_headshots(
    "Path/to/Input/Images",
    "Path/to/sOutput/Images",
    x_offset=-7, 
    y_offset=0
)