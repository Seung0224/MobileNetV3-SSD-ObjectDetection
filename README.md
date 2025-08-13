# **OpenCV SSD Object Detection (Python)**

Windows 환경에서 **OpenCV DNN 모듈**과 **SSD MobileNet v3** 모델을 활용하여  
이미지 속 객체를 실시간으로 탐지하고 시각화하는 Python 프로젝트입니다.  

사전 학습된 **TensorFlow Object Detection** 모델(`.pb`)과 설정 파일(`.pbtxt`)을 사용하며,  
CPU 기반 추론이 가능하도록 구현되었습니다.  

---

<img width="1202" height="732" alt="image" src="https://github.com/user-attachments/assets/c3c7c189-0bf8-43d1-8fe8-1822961b0b26" />


## 📦 프로젝트 개요

- **플랫폼:** Python 3.10+  
- **추론 엔진:** OpenCV DNN (`cv2.dnn_DetectionModel`)  
- **목적:** 이미지 속 COCO 데이터셋 클래스 객체 탐지 및 시각화  
- **모델:** `ssd_mobilenet_v3_large_coco_2020_01_14` (TensorFlow, COCO pre-trained)  
- **결과물:** 탐지된 객체의 바운딩 박스 + 클래스명 + 신뢰도 표시

---

## ✅ 주요 기능

### 1. 📂 모델 로드 & 추론
- 사전 학습된 `.pb` 모델과 `.pbtxt` 설정 파일을 로드
- OpenCV DNN의 `cv2.dnn_DetectionModel` API 사용
- 입력 크기, 정규화, 채널 순서 설정 가능

### 2. 🖼️ 이미지 탐지 & 오버레이
- **COCO 클래스 이름** 매핑 지원
- 신뢰도(Confidence) 임계값 설정 가능 (`confThreshold`)
- 탐지된 객체마다 **Bounding Box**와 클래스명 오버레이

### 3. ⏱ 추론 시간 측정
- Detection 연산 전후 시간을 기록하여 순수 추론 시간(ms) 출력

### 4. 💾 화면 유지 & 시각화
- OpenCV 창을 통해 결과 이미지 시각화
- `cv2.waitKey(0)`를 이용해 사용자가 닫을 때까지 화면 유지

---

## 📂 폴더 구조 예시
