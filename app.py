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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

class ImageRequest(BaseModel):
    imageUrls: List[str]
    strongClustering: Optional[bool] = True
    eyeClosing: Optional[bool] = False
    blurred: Optional[bool] = False

# 공통 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClusteringModel:
    def __init__(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Identity()  # Remove the classification layer
        self.model = self.model.to(device)
        self.threshold = 0.775  # Default threshold

    def set_threshold(self, strong_clustering):
        self.threshold = 0.775 if strong_clustering else 0.675

    def get_image_embedding(self, image):
        img_tensor = preprocess(image).unsqueeze(0).to(device)
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
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)  # Assuming 2 classes: blur and sharp
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.to(device)
 
    def predict_blur(self, image):
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        classes = ['sharp', 'blur']
        predicted_class = classes[predicted.item()]
        return predicted_class == 'blur'

class EyeStateDetector:
    def __init__(self, model_path):
        self.yolo_model = YOLO("models/yolov8/yolov8n-face.pt")
        self.efficientnet_model = self.load_efficientnet_model(model_path)
        self.class_names = ['ClosedFace', 'OpenFace']

    def load_efficientnet_model(self, model_path):
        net = models.efficientnet_b0(pretrained=True)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 2)
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net.to(device)
        net.eval()
        return net

    def face_detection(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(source=rgb_image, conf=0.6, max_det=15, verbose=False)
        faces = []
        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                face_width, face_height = x2 - x1, y2 - y1
                if face_width >= 200 and face_height >= 200:
                    face_image = image[y1:y2, x1:x2]
                    faces.append(face_image)
        return faces

    def eye_detection(self, face_image):
        image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.efficientnet_model(image)
            _, preds = torch.max(outputs, 1)
        return self.class_names[preds.item()]

    def predict(self, image_path):
        image = cv2.imread(image_path)
        faces = self.face_detection(image)
        if not faces:
            return False
        for face in faces:
            eye_state = self.eye_detection(face)
            if eye_state == 'ClosedFace':
                return True
        return False

# FastAPI 엔드포인트
clustering_model = ClusteringModel()
blur_model = BlurModel("models/blurred_0620.pth")
eye_state_detector = EyeStateDetector("models/eyeclosing_0622.pth")

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
                # print(f"Downloaded image: {image_path}")  # 로그 추가
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")  # 로그 추가

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            if request.blurred and blur_model.predict_blur(image):
                os.remove(image_path)
                print(f"Removed blurred image: {image_path}")  # 로그 추가
                continue

            if request.eyeClosing and eye_state_detector.predict(image_path):
                os.remove(image_path)
                print(f"Removed eye-closing image: {image_path}")  # 로그 추가
                continue
        except Exception as e:
            print(f"Failed to process image: {image_path}, Error: {e}")  # 로그 추가

    remaining_images = [image_path for image_path in image_paths if os.path.exists(image_path)]
    clustering_model.set_threshold(request.strongClustering)
    grouped_images = clustering_model.process(remaining_images)

    for i in range(len(grouped_images)):
        for j in range(len(grouped_images[i])):
            url_idx = int(os.path.basename(grouped_images[i][j]).rstrip(".png"))
            grouped_images[i][j] = request.imageUrls[url_idx]

    return {"groupedImages": grouped_images}
