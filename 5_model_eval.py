# BLEU Score로 모델 평가

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


# 텍스트 데이터 불러오기
def load_doc(filename):
    # 파일 열기
    file = open(filename, 'r')
    # text로 모든 내용 읽어오기
    text = file.read()
    # 파일 닫기
    file.close()
    return text


# 데이터셋 불러오기
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # 라인별로 진행
    for line in doc.split('\n'):
        # 빈 라인 스킵
        if len(line) < 1:
            continue
        # 이미지 명 가져오기
        identifier = line.split('.')[0]
        # dataset에 추가
        dataset.append(identifier)
    return set(dataset)


# 2_prep_text 에서 만든 형태소화된 문장과 데이터셋 묶기
def load_clean_descriptions(filename, dataset):
    # 파일 불러오기
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # tokenize
        tokens = line.split()
        # 이미지명과 설명을 분리
        image_id, image_desc = tokens[0], tokens[1:]
        # dataset에 있는 이미지만 처리
        if image_id in dataset:
            # 리스트 생성
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # 설명에 시작점과 종료점 추가
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # 저장
            descriptions[image_id].append(desc)
    return descriptions


# 이미지 특징 불러오기
def load_photo_features(filename, dataset):
    # 1_prep_img 에서 생성한 특징 파일 읽기
    all_features = load(open(filename, 'rb'))
    # dataset에 있는 이미지의 특징만 가져오기
    features = {k: all_features[k] for k in dataset}
    return features


# 설명 리스트화 & 문장화
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# Tokenizer
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    # Tokenizer 불러오기
    tokenizer = Tokenizer()
    # 단어 벡터화
    # 단어의 사용 횟수에 따라 1번부터 번호를 매김
    tokenizer.fit_on_texts(lines)
    return tokenizer


# 설명의 최대길이
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


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


# 모델 평가
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # 모든 데이터 셋
    for key, desc_list in descriptions.items():
        # 캡션 생성
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # true 값과 predict 값 저장
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # BLEU Score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# train dataset으로 학습한 모델을
# 학습하지 않은 test dataset으로 정확도 평가

# train dataset(60%) => 학습시킬 데이터셋
filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# 설명
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# 문장 최대길이
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# test dataset(20%)
filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/test.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# 설명
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# 이미지 특징
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# 4_model_fit 에서 만든 모델들을 불러오기
filename = 'model-ep020-loss4.394-val_loss5.023.h5'
model = load_model(filename)
# 모델 평가
print('Small Data Set Model : ')
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

filename = 'model-ep020-loss1.024-val_loss1.004.h5'
model = load_model(filename)
# 모델 평가
print('\n\nBig Data Set Model : ')
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
