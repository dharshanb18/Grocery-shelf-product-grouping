import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import requests
import os

# Load YOLO model
yolo_model = YOLO(r"shelf images detection\models\epoch15.pt")

#load pretrained cnn model I have downloaded locally I am using resnet
cnn_model = models.resnet50(pretrained=True)
cnn_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_labels = requests.get(LABELS_URL).json()

def get_class_label(output):
    _, predicted = torch.max(output, 1)
    return predicted.item()

def draw_bounding_boxes(image, predictions):
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bounding_box'])
        # Only display the confidence score
        label = f"Confidence: {pred['yolo_confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Display only confidence score
    return image


def process_image_and_save_annotated(image_path):
    results = yolo_model.predict(source=image_path, conf=0.5)

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    detected_products = []

    for result in results:
        boxes = result.boxes  # Get bounding boxes

        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)
            conf = box.conf[0].item()     # Confidence score
            cls = box.cls[0].item()       # YOLO class label

            # Crop the detected object from the image
            x1, y1, x2, y2 = map(int, xyxy)
            cropped_image = image[y1:y2, x1:x2]

            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

            with torch.no_grad():
                output = cnn_model(input_batch)

            cnn_class_index = get_class_label(output)
            cnn_class_name = class_labels[cnn_class_index]  # Get the class name from ImageNet

            product_info = {
                "yolo_class": cls,
                "yolo_confidence": conf,
                "bounding_box": xyxy,
                "cnn_class_index": cnn_class_index,
                "cnn_class_name": cnn_class_name
            }
            detected_products.append(product_info)

    output_image = draw_bounding_boxes(image, detected_products)

    output_dir = r"shelf images detection/annotated"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_image_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_annotated.jpg'))
    cv2.imwrite(output_image_path, output_image)
    print(f"Annotated image saved to {output_image_path}")
    print(os.path.basename(output_image_path))
    return os.path.basename(output_image_path)




#detected_products = process_image_and_save_annotated(r"C:\Users\DHARSHAN BALAJI\Downloads\sample_images\sample_images\2024_01_16_1705384430514.jpg")
