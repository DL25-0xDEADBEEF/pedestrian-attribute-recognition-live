# test.py
import torch

# PyTorch 2.6 호환성을 위해 weights_only=False 추가
checkpoint = torch.load('25.pth.tar', map_location='cpu', weights_only=False)

print("=== 체크포인트 정보 ===")
print("체크포인트 키:", list(checkpoint.keys()))
print()

# 기본 정보들 확인
if 'epoch' in checkpoint:
    print(f"에포크: {checkpoint['epoch']}")
if 'best_accu' in checkpoint:
    print(f"최고 정확도: {checkpoint['best_accu']}")
if 'arch' in checkpoint:
    print(f"Architecture: {checkpoint['arch']}")
if 'dataset' in checkpoint:
    print(f"Dataset: {checkpoint['dataset']}")
if 'experiment' in checkpoint:
    print(f"Experiment: {checkpoint['experiment']}")

print()

# state_dict 정보 확인
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("=== 모델 레이어 정보 ===")
    print(f"총 레이어 수: {len(state_dict)}")
    
    # 마지막 분류기 레이어들 찾기
    classifier_layers = []
    for key in state_dict.keys():
        if 'classifier' in key or 'fc' in key or 'head' in key:
            classifier_layers.append(key)
    
    print("\n분류기 레이어들:")
    for layer in classifier_layers:
        shape = state_dict[layer].shape
        print(f"  {layer}: {shape}")
    
    # 첫 번째와 마지막 몇 개 레이어 표시
    all_keys = list(state_dict.keys())
    print(f"\n첫 5개 레이어:")
    for key in all_keys[:5]:
        print(f"  {key}: {state_dict[key].shape}")
    
    print(f"\n마지막 5개 레이어:")
    for key in all_keys[-5:]:
        print(f"  {key}: {state_dict[key].shape}")

# optimizer 정보 확인
if 'optimizer' in checkpoint:
    print(f"\n=== 옵티마이저 정보 ===")
    optimizer_state = checkpoint['optimizer']
    if 'param_groups' in optimizer_state:
        param_groups = optimizer_state['param_groups']
        if len(param_groups) > 0:
            print(f"학습률: {param_groups[0].get('lr', 'Unknown')}")
            print(f"가중치 감쇠: {param_groups[0].get('weight_decay', 'Unknown')}")
            print(f"모멘텀: {param_groups[0].get('momentum', 'Unknown')}")

# 기타 정보
print(f"\n=== 기타 정보 ===")
for key, value in checkpoint.items():
    if key not in ['state_dict', 'optimizer']:
        if isinstance(value, (int, float, str, bool)):
            print(f"{key}: {value}")
        elif hasattr(value, '__len__'):
            print(f"{key}: (길이 {len(value)})")
        else:
            print(f"{key}: {type(value)}")
