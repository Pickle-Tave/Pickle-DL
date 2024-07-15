import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import faiss
import cv2
import warnings
from torchvision.models import EfficientNet_B2_Weights,MobileNet_V2_Weights 


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

class ImageRequest(BaseModel):
    imageUrls: List[str]
    strongClustering: Optional[bool] = True
    eyeClosing: Optional[bool] = False
    blurred: Optional[bool] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClusteringModel:
    def __init__(self):
        self.model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Identity()  # Remove the classification layer
        self.model = self.model.to(device)
        self.threshold = 0.65  
        self.imagesize = 288
        self.preprocess = self.get_preprocess(self.imagesize)

    def get_preprocess(self, imagesize):
        return transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_threshold(self, strong_clustering):
        self.imagesize = 288 if strong_clustering else 512
        self.preprocess = self.get_preprocess(self.imagesize)

    def get_image_embedding(self, image):
        img_tensor = self.preprocess(image).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            embedding_vector = self.model(img_tensor).cpu().numpy().flatten()
        return embedding_vector

    def process(self, image_paths):
        embeddings = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            embedding = self.get_image_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        if len(embeddings) == 0:
            return []
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        D, I = index.search(embeddings, k=len(embeddings))
        groups = []
        visited = set()
        for i in range(len(embeddings)):
            if i in visited:
                continue
            group = [image_paths[i]]
            visited.add(i)
            for j in range(len(I[i])):
                if D[i][j] >= self.threshold and I[i][j] not in visited:
                    group.append(image_paths[I[i][j]])
                    visited.add(I[i][j])
            groups.append(group)
        
        return groups


class BlurModel:
    def __init__(self, model_path):
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2) 

        blur_state_dict = torch.load(model_path, map_location=device)
        new_blur_state_dict = {}
        for k, v in blur_state_dict.items():
            new_key = k.replace('model.', '')
            new_blur_state_dict[new_key] = v
        self.model.load_state_dict(new_blur_state_dict,strict=True)
        self.model.to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
 
    def predict_blur(self, image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        return predicted.item() == 1

class EyeStateDetector:
    def __init__(self, model_path):
        self.yolo_model = YOLO("models/yolov8/yolov8n-face.pt")
        self.efficientnet_model = self.load_efficientnet_model(model_path)
        self.preprocess = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_efficientnet_model(self, model_path):
        net = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 2)
        net.load_state_dict(torch.load(model_path, map_location=device),strict=True)
        net.to(device)
        net.eval()
        return net

    def face_detection(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(source=rgb_image, conf=0.7, max_det=11, verbose=False)

        total_bbox = sum(len(result.boxes.xyxy) for result in results)
        if total_bbox > 7:
            return []
        
        faces = []
        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                face_width, face_height = x2 - x1, y2 - y1
                if face_width >= 60 and face_height >= 60:
                    face_image = image[y1:y2, x1:x2]
                    faces.append(face_image)
        return faces

    def eye_detection(self, face_image):
        image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        image = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.efficientnet_model(image)
            _, preds = torch.max(outputs, 1)
        return preds

    def predict(self, image_path):
        image = cv2.imread(image_path)
        faces = self.face_detection(image)
        if not faces:
            return False
        for face in faces:
            eye_state = self.eye_detection(face)
            if eye_state == 0:
                return True
        return False

# FastAPI 엔드포인트
clustering_model = ClusteringModel()
blur_model = BlurModel("models/blurred_0711.pth")
eye_state_detector = EyeStateDetector("models/eyeclosing_0714.pth")

@app.post("/images/classify")
def process_urls(request: ImageRequest):
    member_ids = [url.split('/')[3] for url in request.imageUrls]
    if len(set(member_ids)) > 1:
        return {"error": "All images must belong to the same member."}

    member_id = member_ids[0]
    member_folder = f"images/{member_id}"
    os.makedirs(member_folder, exist_ok=True)

    image_paths = []
    for idx, url in enumerate(request.imageUrls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_path = f"{member_folder}/{idx}.png"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                image_paths.append(image_path)
                # print(f"Downloaded image: {image_path}")  
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")  

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            if request.blurred and blur_model.predict_blur(image):
                os.remove(image_path)
                # print(f"Removed blurred image: {image_path}")  
                continue

            if request.eyeClosing and eye_state_detector.predict(image_path):
                os.remove(image_path)
                # print(f"Removed eye-closing image: {image_path}")  
                continue
        except Exception as e:
            print(f"Failed to process image: {image_path}, Error: {e}")  

    remaining_images = [image_path for image_path in image_paths if os.path.exists(image_path)]
    clustering_model.set_threshold(request.strongClustering)
    grouped_images = clustering_model.process(remaining_images)

    for i in range(len(grouped_images)):
        for j in range(len(grouped_images[i])):
            url_idx = int(os.path.basename(grouped_images[i][j]).rstrip(".png"))
            grouped_images[i][j] = request.imageUrls[url_idx]

    shutil.rmtree(member_folder)

    return {"groupedImages": grouped_images}
