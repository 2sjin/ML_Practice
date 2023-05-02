import torch
from torchvision import transforms
from PIL import Image

# 모델 불러오기
model = torch.load('./CNN/model.pth')

# 학습을 위해 필요한 라이브러리를 불러옵니다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

# 데이터셋을 불러옵니다.
# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['김종국', '마동석', '이병헌']
print('클래스:', class_names)

# 이미지 테스트
image = Image.open('./CNN/test_image.jpg')
image = transforms_test(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    print('예측 결과: ' + class_names[preds[0]])