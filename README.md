# Pedestrian Attribute Recognition Live Demo

> **Term Project for Deep Learning Course**  
> School of Computing, Gachon University  
> Spring 2025

Real-time pedestrian attribute recognition system using OAK camera. Recognizes 43 pedestrian attributes (gender, age, clothing, accessories, etc.) in real-time.

실시간 보행자 속성 인식 시스템입니다. OAK 카메라를 통해 실시간으로 보행자의 43가지 속성(성별, 연령, 의류, 액세서리 등)을 인식합니다.

---

## 🎯 Key Features / 주요 기능

- **Real-time Pedestrian Attribute Recognition**: Simultaneous classification of 43 attributes
- **Multi-scale Spatial Transform Network**: High-performance model based on ICCV 2019
- **OAK Camera Support**: Real-time streaming with DepthAI cameras
- **Remote Inference**: Server-client architecture via SSH port forwarding
- **Headless Mode**: Console output support without GUI

---

- **실시간 보행자 속성 인식**: 43개 속성 동시 분류
- **Multi-scale Spatial Transform 네트워크**: ICCV 2019 기반 고성능 모델
- **OAK 카메라 지원**: DepthAI 카메라 실시간 스트리밍
- **원격 추론**: SSH 포트 포워딩을 통한 서버-클라이언트 구조
- **Headless 모드**: GUI 없이 콘솔 출력 지원

## 🎯 Supported Attributes (FootTraffic Dataset) / 지원 속성 (FootTraffic 데이터셋)

### Basic Information / 기본 정보
- Gender / 성별: `male`, `female`
- Age / 연령: `child`, `teenager`, `adult`, `senior`

### Clothing / 의류
- Upper / 상의: `long_sleeve`, `short_sleeve`, `sleeveless`, `onepice`
- Lower / 하의: `long_pants`, `short_pants`, `skirt`
- Colors / 색상: `top_red`, `top_blue`, `bottom_black`, etc.

### Accessories / 액세서리
- `hat`, `glasses`, `bag`, `carrier`, `umbrella`

## 🎓 Academic Information / 학술 정보

This project implements a multi-scale spatial transformer network for pedestrian attribute recognition, based on recent advances in computer vision and deep learning. The model uses ImageNet pre-trained BN-Inception as backbone with custom spatial attention mechanisms optimized for the FootTraffic dataset.

이 프로젝트는 컴퓨터 비전과 딥러닝의 최신 발전을 기반으로 한 보행자 속성 인식을 위한 다중 스케일 공간 변환 네트워크를 구현합니다. 모델은 ImageNet 사전 훈련된 BN-Inception을 백본으로 사용하며, FootTraffic 데이터셋에 최적화된 맞춤형 공간 주의 메커니즘을 포함합니다.

### Key Technical Contributions / 주요 기술적 기여
- Multi-scale spatial attention for attribute-specific localization
- Real-time inference optimization for edge deployment
- Robust streaming pipeline for OAK camera integration

## 📋 System Requirements / 시스템 요구사항

### Hardware / 하드웨어
- **OAK Camera** (OAK-D, OAK-D Lite, OAK-1, etc.)
- **GPU Recommended** (CUDA support, minimum 4GB VRAM)
- **USB 3.0 Port**

### Software / 소프트웨어
- Python 3.8+
- CUDA 11.0+ (for GPU usage)

## 🚀 Installation / 설치 방법

### 1. Clone Repository / 저장소 클론
```bash
git clone https://github.com/DL25-0xDEADBEEF/pedestrian-attribute-recognition-live.git
cd pedestrian-attribute-recognition-live
```

### 2. Create Virtual Environment (Recommended) / 가상환경 생성 (권장)
```bash
conda create -n pedestrian_attr python=3.8
conda activate pedestrian_attr
```

### 3. Install Dependencies / 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model / 사전 훈련된 모델 다운로드
Place the pre-trained model in the project root:
```bash
# Download model file (25.pth.tar)
wget [model_download_link] -O 25.pth.tar
```

사전 훈련된 모델을 프로젝트 루트에 배치:
```bash
# 모델 파일 (25.pth.tar) 다운로드
wget [모델_다운로드_링크] -O 25.pth.tar
```

## 📦 Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
depthai>=2.21.0
Pillow>=8.3.0
numpy>=1.21.0
argparse
```

## 🎮 Usage / 사용 방법

### 📡 Basic Usage (Local) / 기본 사용법 (로컬)

**Step 1: Start Server (Inference Server) / 1단계: 서버 시작 (추론 서버)**
```bash
python server_inference.py --model_path 25.pth.tar --dataset foottraffic --port 9999
```

**Step 2: Start Client (Camera Streaming) / 2단계: 클라이언트 시작 (카메라 스트리밍)**
```bash
python local_camera_stream.py
```

### 🌐 Remote Usage (SSH) / 원격 사용법 (SSH)

**On Server Computer (with GPU) / 서버 컴퓨터에서 (GPU가 있는 머신):**
```bash
# Connect with SSH port forwarding / SSH 포트 포워딩으로 연결
ssh -L 9999:localhost:9999 username@server_ip

# Start server (headless mode recommended) / 서버 시작 (headless 모드 권장)
python server_inference.py --model_path 25.pth.tar --headless
```

**On Client Computer (with OAK camera) / 클라이언트 컴퓨터에서 (OAK 카메라가 연결된 머신):**
```bash
python local_camera_stream.py
```

## ⚙️ Configuration & Customization / 설정 및 커스터마이징

### 1. Network Settings / 네트워크 설정

**Change Port / 포트 변경:**
```bash
# Server / 서버
python server_inference.py --model_path 25.pth.tar --port 8888

# Client (modify local_camera_stream.py) / 클라이언트 (local_camera_stream.py 수정 필요)
streamer = CameraStreamer(host='localhost', port=8888)
```

**Remote IP Connection / 원격 IP 연결:**
```python
# Modify in local_camera_stream.py / local_camera_stream.py에서 수정
streamer = CameraStreamer(host='192.168.1.100', port=9999)  # Server IP / 서버 IP
```

### 2. Camera Settings / 카메라 설정

```python
# Modify in setup_oak_camera function of local_camera_stream.py
# local_camera_stream.py의 setup_oak_camera 함수에서 수정
cam_rgb.setPreviewSize(1280, 720)   # Change resolution / 해상도 변경
cam_rgb.setFps(30)                  # Change FPS / FPS 변경
```

### 3. Inference Settings / 추론 설정

```bash
# Use different dataset model / 다른 데이터셋 모델 사용
python server_inference.py --model_path model.pth --dataset pa100k

# Adjust confidence threshold (modify confidence_threshold variable in code)
# 신뢰도 임계값 조정 (코드 내 confidence_threshold 변수 수정)
```

## 🛠️ Troubleshooting / 문제 해결

### OAK Camera Connection Issues / OAK 카메라 연결 문제
```bash
# Check USB permissions / USB 권한 확인
sudo usermod -a -G dialout $USER

# Test DepthAI / DepthAI 테스트
python -c "import depthai as dai; print('DepthAI installation complete')"
```

### CUDA Issues / CUDA 문제
```bash
# Check CUDA version / CUDA 버전 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU only / CPU만 사용하려면
export CUDA_VISIBLE_DEVICES=""
```

### Network Connection Issues / 네트워크 연결 문제
```bash
# Check port usage / 포트 사용 확인
netstat -tulpn | grep 9999

# Check firewall / 방화벽 확인
sudo ufw allow 9999
```

## 📊 Output Format / 출력 형식

### Console Output Example / 콘솔 출력 예시
```
================================================================================
Frame 1050 | Inference Time: 23.5ms
================================================================================
🔍 Detected Attributes:
   1. 🟢 female               : 0.892
   2. 🟢 adult                : 0.847
   3. 🟡 long_sleeve          : 0.723
   4. 🟡 bag                  : 0.681

📊 Top 5 Probabilities:
  1. ✅ female               : 0.892
  2. ✅ adult                : 0.847
  3. ❌ senior               : 0.789
  4. ✅ long_sleeve          : 0.723
  5. ✅ bag                  : 0.681
================================================================================
```


## 🔧 Advanced Usage / 고급 사용법

### Batch Processing / 배치 처리
```python
# Process multiple images / 여러 이미지 처리
python batch_inference.py --input_dir ./images --output_dir ./results
```

### Performance Benchmark / 성능 벤치마크
```bash
# Measure inference speed / 추론 속도 측정
python benchmark.py --model_path 25.pth.tar --iterations 1000
```

## 📚 References / 참고문헌

- ICCV 2019: "Improving Pedestrian Attribute Recognition with Weakly-supervised Multi-scale Attribute-specific Localization"
- FootTraffic Dataset for Pedestrian Attribute Recognition
- DepthAI Camera SDK Documentation
