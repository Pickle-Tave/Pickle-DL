 # Pickle-DL

Tave 13기 팀 23456 Pickle의 딥러닝 페이지입니다.

## 😃Member

|         | 김지헌    | 백채은    |
|---------|:---------:|:---------:|
|깃허브     |<a href="https://github.com/ben8169"> <img src="https://avatars.githubusercontent.com/ben8169" width="100px;"></a> | <a href="https://github.com/bce5180"> <img src="https://avatars.githubusercontent.com/bce5180" width="100px;"></a>|  

## 🛠️기술 소개
### 1. 유사한 사진 클러스터링
   - EfficientNet b2 + 출력 layer를 Identity Matrix로 변경해 이미지를 임베딩
   - 임베딩 벡터 간의 유사도를 Faiss를 이용하여 측정, 0.65 이상의 유사도를 가진 이미지를 유사하다고 판단
   - Image size를 288로 하여 보다 정확한 분류를, 512로 하여 Feature를 넓게 분석하여 포괄적인 분류를 가능하도록 설계  
![Frame](https://github.com/user-attachments/assets/665f73bb-5e9d-4c9b-803b-a47c42dc0b3c)
  
### 2. 흐릿한 사진 제거
   - Image size를 224로 하여 gray scale로 판단
   - 너무 깊은 Feature까지 학습하지 않도록 가벼운 모델인 MobileNetV2를 채택
   - [Postech RealBLur Dataset](https://cg.postech.ac.kr/research/realblur/), [Discriminative Blur Detection Features](https://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/index.html#downloads) +α, 약 50,000장의 Image를 사용하여 Fine-Tuning
![Frame-1](https://github.com/user-attachments/assets/0a0b0f88-5278-48c7-8ac0-9f249f78f7a6)

     
### 3. 눈 감은 사진 제거
   - [Yolov8](https://github.com/ultralytics/ultralytics)을 사용하여 특정 size 이상의 얼굴만 Crop
   - EfficientNet b2 모델을 채택하여 Closed / Non-Closed(눈뜬 사진을 비롯한 선글라스, 안경, 안대, 눈가림 등) 분류
   - [CEW DATASET](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html), [AiHub Face Parsing 데이터 일부](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71413), Google&Naver Crawling으로 약 8,500장의 Image를 사용하여 Fine-Tuning
![Frame-2](https://github.com/user-attachments/assets/408ce57e-8f9a-4dfa-bddc-f62da4babae0)

  

## 🗂️라이브러리 Version
```bash
fastapi==0.111.0
faiss-cpu==1.8.0
numpy==1.26.4
torch==2.3.1+cu121
Pillow==10.3.0
torchvision==0.18.1+cu121
ultralytics==8.2.46
pydantic==2.7.4
requests==2.25.1
opencv-python-headless==4.10.0
```



##



