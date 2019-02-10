# 이미지 데이터 준비 - 불러오기, 특징 추출, 피클 저장
# Keras를 사용한 이미지 데이터 처리

import os
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


# 특징 추출
def extract_features(directory):
    # 이미지 분석 모델 불러오기
    model = VGG16()
    # 모델 재구성
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # 모델 요약 출력
    print(model.summary())
    # 특징을 저장할 딕셔너리 생성
    features = dict()
    # 진행 상황 출력용 변수
    count = 0
    size = len(next(os.walk(directory))[2])
    for name in listdir(directory):
        # 이미지 경로
        filename = directory + '/' + name
        # VGG16 모델은 224 X 224 픽셀의 이미지를 처리
        image = load_img(filename, target_size=(224, 224))
        # 이미지 픽셀을 numpy 배열로 변경
        image = img_to_array(image)
        # 배열 크기 변경
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess_input() : 이미지를 모델에 필요한 형식으로 변경
        image = preprocess_input(image)
        # predict() : input으로 주어진 이미지의 출력 예측을 생성
        feature = model.predict(image, verbose=0)
        # 이미지 id 분리
        image_id = name.split('.')[0]
        # 이미지 id에 예측 저장
        features[image_id] = feature
        count = count+1
        print('(%d/%d)> %s' %(count, size, name))
    return features


# 경로 설정
directory = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flicker8k_Dataset'
# 모든 이미지로부터 특징 추출
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# 피클 파일로 저장
dump(features, open('features.pkl', 'wb'))
