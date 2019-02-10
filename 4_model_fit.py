# 모델 학습 및 학습된 모델 저장

from numpy import array
from pickle import load
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


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


# 학습 시퀀스 만들기
def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    # key = image_id
    for key, desc_list in descriptions.items():
        # image_id의 desc(설명)
        for desc in desc_list:
            # tokenizer를 통해 문장을 벡터화한 단어들로 바꿈
            # startseq child in pink dress is climbing up set of stairs in an entry way endseq
            #    1      43   4  192  130   7   34      36 193  9  255   4  25  384  194   2
            seq = tokenizer.texts_to_sequences([desc])[0]
            # one-hot encoding
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                # pad_sequences(시퀀스, 최대길이) => 길이가 다른 seq들의 길이를 맞추고
                # 2차원의 numpy 배열로 만들어서 리턴
                # input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # to_categorial(변환할 벡터, 최대 정수)
                # 벡터를 2차원 배열로 만들어서 리턴
                # one-hot encoding
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # 저장
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


# 모델 정의
def define_model(vocab_size, max_length):
    # 특징 추출 모델(CNN)
    # input tensor1
    inputs1 = Input(shape=(4096,))
    # Dropout -> overfitting 방지
    fe1 = Dropout(0.5)(inputs1)
    # Dense(256) => 256개의 hidden unit을 가지는 fully connected layer
    # activation='relu' => 활성화 함수로 Relu 사용
    fe2 = Dense(256, activation='relu')(fe1)
    # 시퀀스 모델(RNN)
    # input tensor2
    inputs2 = Input(shape=(max_length,))
    # Embedding(input_dim, output_dim) => 정수를 고밀도 벡터로 변환
    # mask_zero=True => 입력 값 0이 패딩값인지 여부
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    # LSTM Cell
    se3 = LSTM(256)(se2)
    # decoder 모델
    # CNN모델과 RNN모델 합치기
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    # activation='softmax' => 활성화 함수로 Softmax 사용
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # inputs=[image, seq], outputs=[word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile(optimizer, loss function, metrics) => 모델 구성
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # 모델 요약, 이미지화
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# train dataset(60%) => 학습시킬 데이터셋
filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# train dataset 설명
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# train dataset 이미지 특징
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# tokenizer
tokenizer = create_tokenizer(train_descriptions)
# tokenizer 저장
dump(tokenizer, open('tokenizer.pkl', 'wb'))
# 단어 개수
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# 문장 최대 길이
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# 학습 시퀀스 만들기
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset(20%) => 개발(검증) 데이터셋을 통해 최적의 모델을 찾기
filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/devImages.txt'
dev = load_set(filename)
print('Dataset: %d' % len(dev))
# dev dataset 설명
test_descriptions = load_clean_descriptions('descriptions.txt', dev)
print('Descriptions: test=%d' % len(test_descriptions))
# dev dataset 이미지 특징
test_features = load_photo_features('features.pkl', dev)
print('Photos: test=%d' % len(test_features))
# 테스트 시퀀스 만들기
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# 모델 학습

# 모델 정의
model = define_model(vocab_size, max_length)
# 저장 경로, 이름
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint 호출
# 모델을 저장하면서 loss율 판단
# ModelCheckpoint -> Epoch 00002: val_loss improved from 5.08815 to 5.02380, saving model to model-ep002-loss4.462-val_loss5.024.h5
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
# 모델 학습
######################## epoch 바꿔서
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint],
          validation_data=([X1test, X2test], ytest))
