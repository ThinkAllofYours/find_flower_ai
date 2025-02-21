import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

def get_args():
    # 명령줄 인자 파싱을 위한 설정
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    parser.add_argument('data_dir', type=str, help='학습 데이터와 검증 데이터가 포함된 루트 디렉토리 경로')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                      help='학습 완료 후 모델 가중치와 설정을 저장할 파일 경로')
    parser.add_argument('--arch', type=str, default='vgg16', help='사용할 모델 아키텍처 (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률 (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=4096, help='은닉층의 유닛 수 (default: 4096)')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에포크 수 (default: 5)')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용 여부')
    parser.add_argument('--resume', type=str, 
                      help='체크포인트 파일 경로 (추가 학습용)')
    
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # 데이터 증강 및 정규화 설정
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 무작위 영역 자르기로 과적합 방지
            transforms.RandomHorizontalFlip(),  # 수평 뒤집기로 데이터 다양성 증가
            transforms.ToTensor(),
            # ImageNet 평균 및 표준편차 값으로 정규화
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 데이터셋 로드 (클래스는 하위 디렉토리 구조로 자동 인식)
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }
    
    # 배치 단위 데이터 로딩 설정
    dataloaders = {
        # 학습 데이터는 셔플 적용, 배치 크기 64
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        # 검증 데이터는 셔플 불필요, 배치 크기 32
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    }
    
    return dataloaders, image_datasets

def build_model(arch, hidden_units, checkpoint=None):
    if checkpoint:  # 체크포인트에서 모델 로드
        model = models.__dict__[checkpoint['arch']](pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, checkpoint['hidden_layers'][0])),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),  # 과적합 방지를 위한 드롭아웃
            ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(checkpoint['hidden_layers'][1], 102)),  # 최종 출력 클래스 수: 102 (아마도 Flower102 데이터셋)
            ('output', nn.LogSoftmax(dim=1))  # NLLLoss 사용을 위한 LogSoftmax
        ]))
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    else:  # 새 모델 생성
        # 지원되는 아키텍처 확인
        if arch.lower() in ['vgg16', 'vgg13']:
            model = models.__dict__[arch.lower()](pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Please use 'vgg16' or 'vgg13'")
        
        # 특징 추출기 부분 파라미터 고정 (전이 학습 시 일반적인 방법)
        for param in model.parameters():
            param.requires_grad = False
        
        # 새로운 분류기 구성 (기존 모델의 분류기 대체)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),  # VGG16의 특징맵 출력 크기: 25088
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),  # 과적합 방지를 위한 드롭아웃
            ('fc2', nn.Linear(hidden_units, 1024)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(1024, 102)),  # 최종 출력 클래스 수: 102 (아마도 Flower102 데이터셋)
            ('output', nn.LogSoftmax(dim=1))  # NLLLoss 사용을 위한 LogSoftmax
        ]))
    
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    # 학습 과정 모니터링을 위한 설정
    steps = 0
    print_every = 5  # 매 5회 미니배치 처리마다 검증 수행
    
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            steps += 1
            
            # 데이터를 선택한 디바이스로 이동
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파 + 역전파 + 최적화
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 주기적으로 검증 수행
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()  # 평가 모드로 전환 (드롭아웃 등 비활성화)
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "  # 평균 학습 손실
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "  # 배치 평균 검증 손실
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")  # 배치 평균 정확도
                
                running_loss = 0
                model.train()  # 다시 학습 모드로 복귀

def save_checkpoint(model, image_datasets, save_path, arch, hidden_units):
    # 모델 재학습을 위해 필요한 모든 정보 저장
    checkpoint = {
        'arch': arch,  # 모델 아키텍처 이름
        'input_size': 25088,  # 분류기 입력 크기
        'output_size': 102,  # 출력 클래스 수
        'hidden_layers': [hidden_units, 1024],  # 은닉층 구성 정보
        'state_dict': model.state_dict(),  # 학습된 가중치
        'class_to_idx': image_datasets['train'].class_to_idx  # 클래스-인덱스 매핑 정보
    }
    
    torch.save(checkpoint, save_path)

def main():
    args = get_args()
    
    # GPU 사용 설정
    if args.gpu:
        if torch.backends.mps.is_available():  # Mac의 MPS 디바이스 확인
            device = torch.device("mps")
        elif torch.cuda.is_available():  # NVIDIA GPU 확인
            device = torch.device("cuda")
        else:
            print("GPU가 요청되었지만 사용할 수 없습니다. CPU를 사용합니다.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    # 데이터 로드
    dataloaders, image_datasets = load_data(args.data_dir)
    
    # 체크포인트 로드 또는 새 모델 생성
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model = build_model(checkpoint['arch'], 
                          checkpoint['hidden_layers'][0], 
                          checkpoint)
    else:
        model = build_model(args.arch, args.hidden_units)
    
    # 모델을 디바이스로 이동
    model = model.to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # 체크포인트에서 옵티마이저 상태 로드
    if args.resume and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # 옵티마이저 상태를 디바이스로 이동
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # 모델 학습
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
    # 체크포인트 저장
    save_checkpoint(model, image_datasets, args.save_dir, args.arch, args.hidden_units)
    
    print(f"Model saved to {args.save_dir}")

if __name__ == '__main__':
    main()
