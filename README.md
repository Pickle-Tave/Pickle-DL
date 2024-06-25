# Pickle-DL
Pickle DeepLearning


<h3>6/29 테런데이 </h3>

- 유사도 모델
    - 기본 faiss 유사도 threshold = 0.775
    - 실제 사진 출력됨

- blurring 모델
    - fine-tuning model('blurred_0620.pth)
    - 흐릿한 사진 path 출력

- Eye closing 모델
    - fine-tuning model('eyeclosing_0622.pth')
    - 눈 감았을 경우 True 반환, path 출력