import os
from PIL import Image
from medmnist import BloodMNIST

# --- Configuration ---
OUTPUT_DIR = "sample_images"
CLASS_NAMES = {
    0: 'basophil', 1: 'eosinophil', 2: 'erythroblast',
    3: 'ig', 4: 'lymphocyte', 5: 'monocyte',
    6: 'neutrophil', 7: 'platelet'
}


def extract_sample_images():
    """
    Loads the BloodMNIST test dataset, finds one sample for each class,
    and saves it as a PNG file.
    """
    print("Loading test dataset to find sample images...")
    # We don't need transforms, we want the original images
    test_dataset = BloodMNIST(split='test', download=True, root="./data")

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    found_classes = set()
    for i in range(len(test_dataset)):
        image, label_tuple = test_dataset[i]
        label = label_tuple[0]

        if label not in found_classes:
            class_name = CLASS_NAMES.get(label, f"unknown_{label}")
            file_path = os.path.join(OUTPUT_DIR, f"{class_name}.png")

            # The image is a PIL Image, we can save it directly
            image.save(file_path)

            print(f"  - Saved sample for class '{class_name}' to {file_path}")
            found_classes.add(label)

        # Stop once we have found one sample for each class
        if len(found_classes) == len(CLASS_NAMES):
            break

    print("\nSample image extraction complete.")


if __name__ == "__main__":
    extract_sample_images()
