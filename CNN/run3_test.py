import torch
from torchvision import transforms
from PIL import Image
from tkinter import filedialog

import config


class_names = ['김종국', '마동석', '이병헌']
print('클래스:', class_names)


# ==============================================================================================
# 모델 불러오기
# ==============================================================================================
model = torch.load(config.dir_rel + '/model.pth')


# ==============================================================================================
# device 객체 생성
# ==============================================================================================
device = config.device


# ==============================================================================================
# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
# ==============================================================================================
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==============================================================================================
# 이미지 테스트
# ==============================================================================================
file_types = (('JPEG 파일', '*.jpg;*.jpeg'), ('PNG 파일', '*.png'), ('GIF 파일', '*.gif'), ('BMP 파일', '*.bmp'))
file_path = filedialog.askopenfilename(initialdir='./', title='이미지 파일 선택', filetypes=file_types)

image = Image.open(file_path)
image = transforms_test(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    print('예측 결과: ' + class_names[preds[0]])