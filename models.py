import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models
from ultralytics import YOLO
import faiss
import cv2


app = FastAPI()

class PresignedURLRequest(BaseModel):
    urls: List[str]
    memberid: str
    eyeclosing: Optional[bool] = False
    blurred: Optional[bool] = False

# 공통 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClusteringModel:
    def __init__(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Identity()  # Remove the classification layer
        self.model = self.model.to(device)
    
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
        threshold = 0.775
        groups = []
        visited = set()
        for i in range(len(embeddings)):
            if i in visited:
                continue
            group = [image_paths[i]]
            visited.add(i)
            for j in range(1, len(I[i])):
                if D[i][j] >= threshold and I[i][j] not in visited:
                    group.append(image_paths[I[i][j]])
                    visited.add(I[i][j])
            groups.append(group)
        
        return groups

class BlurModel:
    def __init__(self, model_path):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)  # Assuming 2 classes: blur and sharp
        self.model.load_state_dict(torch.load(model_path))
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO("picklework/eyeclosing/yolov8n-face.pt")
        self.efficientnet_model = self.load_efficientnet_model(model_path)
        self.class_names = ['ClosedFace', 'OpenFace']

    def load_efficientnet_model(self, model_path):
        net = models.efficientnet_b0(pretrained=True)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 2)
        net.load_state_dict(torch.load(model_path))
        net.to(self.device)
        net.eval()
        return net

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)
        return image

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
        image = self.preprocess_image(face_image).to(self.device)
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
blur_model = BlurModel('blurred_0620.pth')
eye_state_detector = EyeStateDetector(r"C:\Users\ben81\GitHub\Pickle-DL\picklework\eyeclosing\models\b0_3rd_rgbfinal_model.pth")

@app.post("/process-urls")
def process_urls(request: PresignedURLRequest):
    member_folder = os.path.join("images", request.memberid)
    os.makedirs(member_folder, exist_ok=True)

    image_paths = []
    for url in request.urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(member_folder, 'wb') as f:
                    f.write(response.content)
            image_paths.append(image_path)
        except Exception as e:
            # print(f"Error downloading {url}: {e}")
            pass

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')

        if request.blurred and blur_model.predict_blur(image):
            os.remove(image_path)
            continue

        if request.eyeclosing and eye_state_detector.predict(image_path):
            os.remove(image_path)
            continue

    remaining_images = [image_path for image_path in image_paths if os.path.exists(image_path)]
    grouped_images = clustering_model.process(remaining_images)

    return {"grouped_images": grouped_images}

# 실행 방법
# uvicorn main:app --reload
