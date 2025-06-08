# local_camera_stream.py
import depthai as dai
import cv2
import socket
import pickle
import struct
import numpy as np
import time

class CameraStreamer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        
    def setup_oak_camera(self):
        """OAK 카메라 설정"""
        pipeline = dai.Pipeline()
        
        # Color 카메라 노드 생성
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(1280, 720)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)
        # cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)  # BGR로 변경

        
        # 출력을 위한 XLinkOut 노드 생성
        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam_rgb.preview.link(xout.input)
        
        return pipeline
    
    def stream_video(self):
        """비디오 스트리밍"""
        print("서버 연결을 시도합니다...")
        
        # 서버 연결 재시도 로직
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 소켓 설정
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)  # 10초 타임아웃
                sock.connect((self.host, self.port))
                print(f"서버 연결 성공: {self.host}:{self.port}")
                break
            except Exception as e:
                retry_count += 1
                print(f"연결 실패 ({retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("5초 후 재시도...")
                    time.sleep(5)
                else:
                    print("서버 연결에 실패했습니다. SSH 포트 포워딩이 설정되어 있는지 확인하세요.")
                    return
        
        # OAK 카메라 설정
        pipeline = self.setup_oak_camera()
        
        try:
            with dai.Device(pipeline) as device:
                print("OAK 카메라 연결 완료")
                video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                print("비디오 스트리밍 시작 (q를 눌러 종료)")
                
                frame_count = 0
                start_time = time.time()
                
                while True:
                    try:
                        # 프레임 가져오기
                        in_frame = video_queue.get()
                        frame = in_frame.getCvFrame()
                        
                        # BGR로 변환 (OpenCV 호환)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # 프레임 크기 조정 (네트워크 대역폭 절약)
                        frame_resized = cv2.resize(frame_bgr, (640, 480))
                        
                        # 프레임 압축 및 직렬화
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        result, encimg = cv2.imencode('.jpg', frame_resized, encode_param)
                        
                        if result:
                            data = pickle.dumps(encimg)
                            size = len(data)
                            
                            # 크기 정보 먼저 전송 (8바이트 고정)
                            sock.sendall(struct.pack("Q", size))
                            # 프레임 데이터 전송
                            sock.sendall(data)
                            
                            frame_count += 1
                            
                            # FPS 계산 및 표시
                            if frame_count % 30 == 0:
                                elapsed_time = time.time() - start_time
                                fps = frame_count / elapsed_time
                                print(f"전송 FPS: {fps:.1f}")
                        
                        # 로컬 미리보기 (선택사항)
                        cv2.putText(frame_resized, f"Streaming to Server", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_resized, f"Frames: {frame_count}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow("Local Camera (Streaming)", frame_resized)
                        
                        # 종료 조건
                        if cv2.waitKey(1) == ord('q'):
                            print("사용자가 종료를 요청했습니다.")
                            break
                            
                    except socket.error as e:
                        print(f"네트워크 오류: {e}")
                        break
                    except Exception as e:
                        print(f"프레임 처리 오류: {e}")
                        continue
                        
        except Exception as e:
            print(f"카메라 연결 실패: {e}")
            print("OAK 카메라가 USB에 연결되어 있는지 확인하세요.")
        finally:
            try:
                sock.close()
            except:
                pass
            cv2.destroyAllWindows()
            print("스트리밍 종료")

if __name__ == "__main__":
    print("=== OAK Camera Streamer ===")
    print("사용법:")
    print("1. SSH 포트 포워딩 설정: ssh -L 9999:localhost:9999 username@server_ip")
    print("2. 서버에서 server_inference.py 실행")
    print("3. 이 스크립트 실행")
    print()
    
    # localhost:9999로 연결 (SSH 포트 포워딩 사용)
    streamer = CameraStreamer(host='localhost', port=9999)
    streamer.stream_video()