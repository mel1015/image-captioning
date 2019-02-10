# 새로운 이미지의 캡션 생성

from pickle import load
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax


# 특징 추출
def extract_features(filename):
    # 이미지 분석 모델 불러오기
    model = VGG16()
    # 모델 재구성
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # 이미지 불러오기
    image = load_img(filename, target_size=(224, 224))
    # 이미지 픽셀을 numpy 배열로 변경
    image = img_to_array(image)
    # 배열 크기 변경
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # 이미지 처리
    image = preprocess_input(image)
    # 특징 추출
    feature = model.predict(image, verbose=0)
    return feature


# 벡터화된 단어를 실제 단어로 변환
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# 이미지 캡션 생성
def generate_desc(model, tokenizer, photo, max_length):
    # 시작 시퀀스
    in_text = 'startseq'
    # 시퀀스의 전체 길이를 반복
    for i in range(max_length):
        # sequence 벡터화
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # input의 길이맞추기(padding)
        sequence = pad_sequences([sequence], maxlen=max_length)
        # 다음 단어 예측
        yhat = model.predict([photo, sequence], verbose=0)
        # 예측한 결과값의 가장 큰 값을 통해
        # 이미지에 나와있을 가장 큰 가능성을 가진 단어를 찾음
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        # 매핑되는 단어가 없을때
        if word is None:
            break
        # 문장에 다음 단어 추가
        in_text += ' ' + word
        # 종료 시퀀스
        if word == 'endseq':
            break
    return in_text


# tokenizer 불러오기
tokenizer = load(open('tokenizer.pkl', 'rb'))
# 모델이 만들 수 있는 최대 문장 길이
max_length = 23
# 모델 불러오기
# 100 Dataset Model
# model = load_model('model-ep020-loss4.394-val_loss5.023.h5')
# 8000 Dataset Model
model = load_model('model-ep020-loss1.024-val_loss1.004.h5')

# 캡셔닝할 이미지
photo1 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/1.jpg')
photo2 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/2.jpg')
photo3 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/3.jpg')
photo4 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/4.jpg')
photo5 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/5.jpg')
photo6 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/6.jpg')
photo7 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/7.jpg')
photo8 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/8.jpg')
photo9 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/9.jpg')
photo10 = extract_features('/Users/sks10/Desktop/Python/Image_Captioning/100dataset/For_Captioning/10.jpg')

# 캡셔닝
description1 = generate_desc(model, tokenizer, photo1, max_length)
print('1.jpg caption: ' + description1)
description2 = generate_desc(model, tokenizer, photo2, max_length)
print('2.jpg caption: ' + description2)
description3 = generate_desc(model, tokenizer, photo3, max_length)
print('3.jpg caption: ' + description3)
description4 = generate_desc(model, tokenizer, photo4, max_length)
print('4.jpg caption: ' + description4)
description5 = generate_desc(model, tokenizer, photo5, max_length)
print('5.jpg caption: ' + description5)
description6 = generate_desc(model, tokenizer, photo6, max_length)
print('6.jpg caption: ' + description6)
description7 = generate_desc(model, tokenizer, photo7, max_length)
print('7.jpg caption: ' + description7)
description8 = generate_desc(model, tokenizer, photo8, max_length)
print('8.jpg caption: ' + description8)
description9 = generate_desc(model, tokenizer, photo9, max_length)
print('9.jpg caption: ' + description9)
description10 = generate_desc(model, tokenizer, photo10, max_length)
print('10.jpg caption: ' + description10)
