import os

# 현재 디렉토리
dir_abs = os.path.dirname(os.path.abspath(__file__))    # 절대 경로
dir_rel = './' + os.path.basename(dir_abs)              # 상대 경로

# 데이터셋 디렉토리
dir_train = dir_abs + '/custom_dataset/train/'
dir_test = dir_abs + '/custom_dataset/test/'