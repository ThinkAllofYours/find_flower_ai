import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
from collections import OrderedDict
import os
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Flower Classification Prediction")
    parser.add_argument('input', type=str, help='입력 이미지 경로')
    parser.add_argument('checkpoint', type=str, help='모델 체크포인트 경로')
    parser.add_argument('--top_k', type=int, default=5, help='상위 K개 예측 결과 출력 (default: 5)')
    parser.add_argument('--category_names', type=str, help='카테고리 이름 매핑 JSON 파일')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용 여부')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(checkpoint['hidden_layers'][1], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    # PIL 이미지 로드
    img = Image.open(image_path)
    
    # 전처리 변환 정의
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 이미지 전처리
    img_tensor = preprocess(img)
    
    return img_tensor

def predict(image_path, model, device, topk=5):
    try:
        model.eval()
        
        # 이미지 전처리
        img_tensor = process_image(image_path)
        
        # 배치 차원 추가 및 device로 이동
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.forward(img_tensor)
            ps = torch.exp(output)
            top_p, top_indices = ps.topk(topk, dim=1)
            
            # CPU로 이동하고 numpy 배열로 변환
            top_p = top_p.cpu().numpy().squeeze()
            top_indices = top_indices.cpu().numpy().squeeze()
            
            # 인덱스를 클래스로 매핑
            idx_to_class = {val: key for key, val in model.class_to_idx.items()}
            top_classes = [idx_to_class[idx] for idx in top_indices]
        
        return top_p, top_classes
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        return None, None

def main():
    args = get_args()
    
    try:
        # 입력 파일 존재 확인
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {args.input}")
        
        # 체크포인트 파일 존재 확인
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        
        # 디바이스 설정
        if args.gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("GPU를 사용합니다.")
            else:
                print("GPU가 요청되었지만 사용할 수 없습니다. CPU를 사용합니다.")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            print("CPU를 사용합니다.")
        
        # 카테고리 이름 로드
        cat_to_name = None
        if args.category_names:
            if not os.path.exists(args.category_names):
                print(f"경고: 카테고리 파일을 찾을 수 없습니다: {args.category_names}")
            else:
                with open(args.category_names, 'r') as f:
                    cat_to_name = json.load(f)
        
        # 모델 로드
        print("모델을 로드하는 중...")
        model = load_checkpoint(args.checkpoint)
        model.to(device)
        
        # 예측 수행
        print("예측을 수행하는 중...")
        probs, classes = predict(args.input, model, device, args.top_k)
        
        if probs is not None and classes is not None:
            # 결과 출력
            print("\n예측 결과:")
            for i, (prob, class_idx) in enumerate(zip(probs, classes), 1):
                if cat_to_name:
                    class_name = cat_to_name.get(class_idx, f"Unknown ({class_idx})")
                else:
                    class_name = f"Class {class_idx}"
                print(f"{i}. {class_name}: {prob*100:.2f}%")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
