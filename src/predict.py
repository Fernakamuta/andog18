import os
import cv2


def load_images_from_folder(folder):
    images = []
    labels = []

    for label in ["0", "1"]:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images.append(img)
                labels.append(int(label))

    return images, labels


folder = "src/laila_images"
images, labels = load_images_from_folder(folder)
