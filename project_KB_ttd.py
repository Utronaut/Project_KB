import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def get_signatures_and_labels(main_path):
    """
    Extracts signatures and labels from the dataset folder structure.
    
    Args:
        main_path (str): Path to the main dataset folder.

    Returns:
        signatures (list): List of processed signature images.
        labels (list): Corresponding labels for the signatures.
        label_names (dict): Mapping of label indices to folder names.
    """
    signatures = []
    labels = []
    label_names = {}
    current_label = 0

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Read the image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Preprocess the signature image
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                resized = cv2.resize(binary, (150, 150))

                # Save the processed signature and label
                signatures.append(resized)
                labels.append(current_label)

            current_label += 1

    return signatures, labels, label_names

# Path to the dataset
main_dataset_path = "C:\\Users\\HP\\Downloads\\DATASET-KB-KELAS-B-20241219T234648Z-001\\DATASET-KB-KELAS-B"

# Load signatures and labels
signatures, labels, label_names = get_signatures_and_labels(main_dataset_path)

# Check if dataset is not empty
if len(signatures) > 0:
    # Flatten the signature images for SVM training
    signatures_flattened = [signature.flatten() for signature in signatures]

    # Encode the labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(signatures_flattened, labels_encoded, test_size=0.2, random_state=42)

    # Train the SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model and label encoder
    joblib.dump(clf, 'svm_signature_model.pkl')
    joblib.dump(le, 'label_encoder_signature.pkl')
else:
    print("Dataset is empty or invalid. Ensure the dataset contains valid signature images.")
    exit()

def upload_and_recognize_signature(image_path, model_path='svm_signature_model.pkl', encoder_path='label_encoder_signature.pkl'):
    """
    Recognizes the signature in a given image using the trained model.

    Args:
        image_path (str): Path to the signature image.
        model_path (str): Path to the saved SVM model.
        encoder_path (str): Path to the saved label encoder.

    Returns:
        None
    """
    # Load the trained model and label encoder
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    # Read and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Invalid image. Ensure the file path is correct.")
        return

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary, (150, 150))
    signature_flattened = resized.flatten().reshape(1, -1)

    # Predict the identity of the signature
    label_encoded = clf.predict(signature_flattened)
    proba = clf.predict_proba(signature_flattened)
    confidence = np.max(proba) * 100  # Convert confidence to percentage

    # Decode the label
    label = le.inverse_transform(label_encoded)[0]
    name = label_names.get(label, "Unknown")

    print(f"Detected signature: {name} with confidence: {confidence:.2f}%")

# Example usage
image_path = input("Enter the path to the signature image: ")
upload_and_recognize_signature(image_path)