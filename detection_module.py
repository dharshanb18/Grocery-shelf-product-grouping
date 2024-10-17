import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import requests

#this is my trained model file after 15 epochs which ran around 4 hours in gpu
yolo_model = YOLO(r"shelf images detection\models\epoch15.pt")

cnn_model = models.resnet50(pretrained=True)
cnn_model.eval()

# Preprocessing for CNN model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#this is the imagenet url
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_labels = requests.get(LABELS_URL).json()

def get_class_label(output):
    _, predicted = torch.max(output, 1)
    return predicted.item()

def draw_bounding_boxes(image, predictions):
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bounding_box'])
        label = f"{pred['cnn_class_name']}: {pred['yolo_confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def predict_products(image_path, save_path=None):
    image = cv2.imread(image_path)
    results = yolo_model.predict(source=image_path, conf=0.5)

    detected_products = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            x1, y1, x2, y2 = map(int, xyxy)
            cropped_image = image[y1:y2, x1:x2]
            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = cnn_model(input_batch)

            cnn_class_index = get_class_label(output)
            cnn_class_name = class_labels[cnn_class_index]

            product_info = {
                "yolo_class": cls,
                "yolo_confidence": conf,
                "bounding_box": xyxy,
                "cnn_class_name": cnn_class_name
            }
            detected_products.append(product_info)

    if save_path:
        output_image = draw_bounding_boxes(image, detected_products)
        cv2.imwrite(save_path, output_image)

    return detected_products
