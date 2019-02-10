# 모델 학습 준비 -> train dataset 생성, 이미지와 캡션 묶기

from pickle import load


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


# training dataset(60%)
filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# 설명 불러오기
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# 특징 불러오기
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
