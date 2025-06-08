# server_inference.py
import socket
import pickle
import struct
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import threading
import queue
import time

# 모델 import
sys.path.append('/home/jhlee/pjt/rootfs/workspace/iccv19_attribute/model')  # 실제 경로로 수정
from inception_iccv import inception_iccv

class InferenceServer:
    def __init__(self, model_path, dataset='foottraffic', port=9999, headless=False):
        self.port = port
        self.model_path = model_path
        self.dataset = dataset
        
        # 속성 정보 설정
        self.attr_nums = {
            'pa100k': 26, 'rap': 51, 'peta': 35, 'foottraffic': 43
        }
        
        # foottraffic 데이터셋 속성 설명 (실제 속성에 맞게 수정하세요)
        self.descriptions = {
            'foottraffic': [
                'male', 'female', 'child', 'teenager', 'adult', 'senior',
                'long_sleeve', 'short_sleeve', 'sleeveless', 'onepice',
                'top_red', 'top_orange', 'top_yellow', 'top_green', 'top_blue',
                'top_purple', 'top_pink', 'top_brown', 'top_white', 'top_grey', 'top_black',
                'long_pants', 'short_pants', 'skirt', 'bottom_type_none',
                'bottom_red', 'bottom_orange', 'bottom_yellow', 'bottom_green', 'bottom_blue',
                'bottom_purple', 'bottom_pink', 'bottom_brown', 'bottom_white', 'bottom_grey', 'bottom_black',
                'carrier', 'umbrella', 'bag', 'hat', 'glasses', 'acc_none', 'pet'
            ],
            'pa100k': [
                'Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back',
                'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
                'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
                'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'
            ]
        }
        
        self.num_classes = self.attr_nums[dataset]
        self.attr_descriptions = self.descriptions.get(dataset, [f'attr_{i}' for i in range(self.num_classes)])
        
        # GPU/CPU 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"사용 디바이스: {self.device}")
        
        # 모델 로드
        self.model = self.load_model()
        
        # 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 멀티스레딩을 위한 큐
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # 성능 측정
        self.frame_count = 0
        self.inference_times = []

        self.headless = True  # GUI 사용 여부
        
    def load_model(self):
        """모델 로드"""
        try:
            model = inception_iccv(pretrained=False, num_classes=self.num_classes)
            
            if os.path.exists(self.model_path):
                print(f"모델 로드 시작: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # 체크포인트에서 state_dict 추출
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"에포크: {checkpoint.get('epoch', 'Unknown')}")
                    print(f"최고 정확도: {checkpoint.get('best_accu', 'Unknown')}")
                else:
                    state_dict = checkpoint
                    
                # DataParallel 모듈 접두사 제거
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                        
                model.load_state_dict(new_state_dict)
                print("✓ 학습된 가중치 로드 완료")
            else:
                print(f"⚠ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                print("Pretrained ImageNet weights만 사용합니다.")
                
            model.to(self.device)
            model.eval()
            print(f"✓ 모델이 {self.device}에 로드되었습니다.")
            return model
            
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            raise
    
    def preprocess_image(self, cv_image):
        """이미지 전처리"""
        try:
            # BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 전처리 적용
            tensor_image = self.transform(pil_image).unsqueeze(0)
            return tensor_image.to(self.device)
        except Exception as e:
            print(f"전처리 오류: {e}")
            return None
    
    def predict(self, image_tensor):
        """속성 예측"""
        try:
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(image_tensor)
                
                # 다중 출력 처리
                if isinstance(output, (tuple, list)):
                    final_output = output[0]
                    for i in range(1, len(output)):
                        final_output = torch.max(final_output, output[i])
                else:
                    final_output = output
                    
                # 확률 계산
                probabilities = torch.sigmoid(final_output).cpu().numpy()[0]
                predictions = (probabilities > 0.5).astype(int)
                
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                return predictions, probabilities, inference_time
                
        except Exception as e:
            print(f"추론 오류: {e}")
            return None, None, 0
    
    def inference_worker(self):
        """추론 워커 스레드"""
        print("추론 워커 스레드 시작")
        
        while True:
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:  # 종료 신호
                    break
                    
                frame, timestamp = item
                
                # 전처리
                image_tensor = self.preprocess_image(frame)
                if image_tensor is None:
                    continue
                    
                # 추론
                predictions, probabilities, inference_time = self.predict(image_tensor)
                if predictions is not None:
                    self.result_queue.put((frame, predictions, probabilities, timestamp, inference_time))
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"워커 스레드 오류: {e}")
    
    def draw_predictions(self, image, predictions, probabilities, inference_time, confidence_threshold=0.6):
        """예측 결과를 이미지에 그리기"""
        h, w = image.shape[:2]
        
        # 반투명 배경
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, min(h-10, 400)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 제목
        cv2.putText(image, "Pedestrian Attributes (Server)", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 성능 정보
        avg_inference_time = np.mean(self.inference_times[-30:]) if self.inference_times else 0
        cv2.putText(image, f"Inference: {inference_time*1000:.1f}ms (avg: {avg_inference_time*1000:.1f}ms)", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(image, f"Device: {self.device.upper()}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 속성 예측 결과
        y_offset = 110
        line_height = 18
        
        # 높은 신뢰도의 속성만 선택
        positive_attrs = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1 and prob > confidence_threshold:
                attr_name = self.attr_descriptions[i]
                positive_attrs.append((attr_name, prob))
        
        # 신뢰도 순으로 정렬
        positive_attrs.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 속성들 표시 (최대 12개)
        max_attrs = min(12, len(positive_attrs))
        if max_attrs == 0:
            cv2.putText(image, "No significant attributes detected", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        else:
            for i, (attr_name, prob) in enumerate(positive_attrs[:max_attrs]):
                # 신뢰도에 따른 색상 변경
                if prob > 0.8:
                    color = (0, 255, 0)  # 초록 (높은 신뢰도)
                elif prob > 0.7:
                    color = (0, 255, 255)  # 노랑 (중간 신뢰도)
                else:
                    color = (0, 165, 255)  # 주황 (낮은 신뢰도)
                
                text = f"{attr_name}: {prob:.3f}"
                cv2.putText(image, text, (20, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def start_server(self):
        """서버 시작"""
        print(f"=== Pedestrian Attribute Inference Server ===")
        print(f"포트: {self.port}")
        print(f"데이터셋: {self.dataset} ({self.num_classes} attributes)")
        print(f"모델: {self.model_path}")
        print()
        
        # 추론 워커 스레드 시작
        inference_thread = threading.Thread(target=self.inference_worker)
        inference_thread.daemon = True
        inference_thread.start()
        
        # 소켓 서버 설정
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(('localhost', self.port))  # localhost로 바인딩
            sock.listen(1)
            print(f"서버 대기 중... 포트 {self.port}")
            print("클라이언트 연결 대기중...")
            print("SSH 포트 포워딩: ssh -L 9999:localhost:9999 username@server_ip")
            print()
            
            conn, addr = sock.accept()
            print(f"✓ 클라이언트 연결됨: {addr}")
            
            data = b""
            payload_size = struct.calcsize("Q")
            
            start_time = time.time()
            received_frames = 0
            
            while True:
                try:
                    # 프레임 크기 수신
                    while len(data) < payload_size:
                        packet = conn.recv(4*1024)
                        if not packet:
                            print("클라이언트 연결 종료")
                            return
                        data += packet
                    
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]
                    
                    # 프레임 데이터 수신
                    while len(data) < msg_size:
                        packet = conn.recv(4*1024)
                        if not packet:
                            print("클라이언트 연결 종료")
                            return
                        data += packet
                    
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    
                    # 프레임 디코딩
                    frame_buffer = pickle.loads(frame_data)
                    frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        received_frames += 1
                        timestamp = time.time()
                        
                        # 추론 큐에 추가 (큐가 가득 차면 오래된 프레임 제거)
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.frame_queue.put((frame, timestamp))
                        
                        # 결과 표시
                        display_frame = frame.copy()

                        # try:
                        #     result_frame, predictions, probabilities, result_timestamp, inference_time = self.result_queue.get_nowait()
                        #     display_frame = self.draw_predictions(result_frame, predictions, probabilities, inference_time)
                        # except queue.Empty:
                        #     # 결과가 없으면 원본 프레임에 대기 메시지 표시
                        #     cv2.putText(display_frame, "Processing...", (20, 50), 
                        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                            # 수정된 코드:
                        try:
                            result_frame, predictions, probabilities, result_timestamp, inference_time = self.result_queue.get_nowait()
                            
                            # 추론 결과를 콘솔에 자세히 출력
                            self.print_detailed_predictions(predictions, probabilities, inference_time, received_frames)
                            
                            # GUI가 활성화된 경우에만 화면 표시
                            if not self.headless:
                                display_frame = self.draw_predictions(result_frame, predictions, probabilities, inference_time)
                                cv2.imshow("Server Inference Result", display_frame)
                                if cv2.waitKey(1) == ord('q'):
                                    print("서버 종료 요청")
                                    break
                                    
                        except queue.Empty:
                            # 30프레임마다 상태 출력
                            if received_frames % 30 == 0:
                                print(f"[INFO] 프레임 {received_frames} 수신 완료 - 추론 대기 중...")


                        # FPS 계산
                        if received_frames % 30 == 0:
                            elapsed = time.time() - start_time
                            fps = received_frames / elapsed
                            print(f"수신 FPS: {fps:.1f}, 총 프레임: {received_frames}")
                        
                        # 화면에 표시
                        # cv2.imshow("Server Inference Result", display_frame)
            # GUI 사용 여부 제어
                        if not self.headless:
                            cv2.imshow("Server Inference Result", display_frame)
                            if cv2.waitKey(1) == ord('q'):
                                print("서버 종료 요청")
                                break
                        else:
                            # 콘솔에 결과 출력
                            if received_frames % 30 == 0:  # 30프레임마다 출력
                                print(f"프레임 {received_frames} 처리 완료")
                                # 예측 결과가 있으면 출력
                                try:
                                    result_frame, predictions, probabilities, result_timestamp, inference_time = self.result_queue.get_nowait()
                                    positive_attrs = []
                                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                                        if pred == 1 and prob > 0.6:
                                            attr_name = self.attr_descriptions[i]
                                            positive_attrs.append((attr_name, prob))
                                    
                                    positive_attrs.sort(key=lambda x: x[1], reverse=True)
                                    print(f"감지된 속성: {', '.join([f'{name}({prob:.2f})' for name, prob in positive_attrs[:5]])}")
                                except:
                                    pass
                        
                        if cv2.waitKey(1) == ord('q'):
                            print("서버 종료 요청")
                            break
                            
                except Exception as e:
                    print(f"프레임 처리 오류: {e}")
                    continue
                    
        except Exception as e:
            print(f"서버 오류: {e}")
        finally:
            # 정리
            self.frame_queue.put(None)  # 워커 스레드 종료 신호
            try:
                conn.close()
            except:
                pass
            sock.close()
            cv2.destroyAllWindows()
            print("서버 종료")

    def print_detailed_predictions(self, predictions, probabilities, inference_time, frame_num, confidence_threshold=0.5):
        """상세한 예측 결과를 콘솔에 출력"""

        # 모든 예측된 속성 수집
        all_predictions = []
        positive_predictions = []

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            attr_name = self.attr_descriptions[i]
            all_predictions.append((attr_name, pred, prob))

            if pred == 1 and prob > confidence_threshold:
                positive_predictions.append((attr_name, prob))

        # 결과 출력
        print(f"\n{'='*80}")
        print(f"프레임 {frame_num} | 추론 시간: {inference_time*1000:.1f}ms")
        print(f"{'='*80}")

        if positive_predictions:
            # 신뢰도 순으로 정렬
            positive_predictions.sort(key=lambda x: x[1], reverse=True)

            print("🔍 감지된 속성:")
            for i, (attr_name, prob) in enumerate(positive_predictions, 1):
                confidence_level = "🟢" if prob > 0.8 else "🟡" if prob > 0.6 else "🟠"
                print(f"  {i:2d}. {confidence_level} {attr_name:<20} : {prob:.3f}")
        else:
            print("❌ 신뢰도가 높은 속성이 감지되지 않았습니다.")

        # 상위 5개 확률 표시 (예측 여부와 관계없이)
        print(f"\n📊 상위 5개 확률:")
        top_probs = sorted(all_predictions, key=lambda x: x[2], reverse=True)[:5]
        for i, (attr_name, pred, prob) in enumerate(top_probs, 1):
            status = "✅" if pred == 1 else "❌"
            print(f"  {i}. {status} {attr_name:<20} : {prob:.3f}")

        print(f"{'='*80}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pedestrian Attribute Inference Server')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='foottraffic', 
                       choices=['foottraffic', 'pa100k', 'rap', 'peta'],
                       help='Dataset type (default: foottraffic)')
    parser.add_argument('--port', type=int, default=9999, 
                       help='Server port (default: 9999)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI (recommended for SSH)')
    
    args = parser.parse_args()
    
    # 서버 시작
    server = InferenceServer(args.model_path, args.dataset, args.port)
    server.start_server()
