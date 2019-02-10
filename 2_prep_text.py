# 텍스트 데이터 준비 - 설명 불러오기, 라인 별로 나누기, 필요 없는 문자 지우기

import string


# 텍스트 데이터 불러오기
def load_doc(filename):
    # 파일 열기
    file = open(filename, 'r')
    # text로 모든 내용 읽어오기
    text = file.read()
    # 파일 닫기
    file.close()
    return text


# 이미지 별로 캡션 저장
def load_descriptions(doc):
    # 딕셔너리 생성
    mapping = dict()
    # 라인 별로 나누기
    for line in doc.split('\n'):
        # tokenize -> 문장을 토큰으로 나누기
        tokens = line.split()
        if len(line) < 2:
            continue
        # token의 첫번째는 image id, 나머지가 설명
        image_id, image_desc = tokens[0], tokens[1:]
        # 파일 형식(.jpg)를 지우고 이미지 이름만 저장
        image_id = image_id.split('.')[0]
        # tokenize 된 설명을 문장으로 바꿔 저장
        image_desc = ' '.join(image_desc)
        # 리스트 생성
        if image_id not in mapping:
            mapping[image_id] = list()
        # 설명 저장
        mapping[image_id].append(image_desc)
    return mapping


# 구두점, 숫자 제거
def clean_descriptions(descriptions):
    # 구두점 제거를 위한 테이블
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize -> 형태소(뜻을 가진 최소 단위)로 나누기
            desc = desc.split()
            # 소문자로 바꾸기
            desc = [word.lower() for word in desc]
            # 캡션으로부터 구두점 제거
            desc = [w.translate(table) for w in desc]
            # 형태소로 나누기위해 1글자 이상만 단어로 인식
            desc = [word for word in desc if len(word) > 1]
            # 알파벳으로 구성된 단어만 인식 -> 숫자 제거
            desc = [word for word in desc if word.isalpha()]
            # 문자열로 저장
            desc_list[i] = ' '.join(desc)


# 설명을 단어 어휘로 변경
def to_vocabulary(descriptions):
    # 설명 문자열을 리스트화
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# 설명 저장
def save_descriptions(descriptions, filename):
    # 설명을 라인별로 나누기
    lines = list()
    # 이미지 명과 설명 합치기
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


filename = '/Users/sks10/Desktop/Python/Image_Captioning/100dataset/Flickr8k_text/Flickr8k.token.txt'
# 설명 불러오기
doc = load_doc(filename)
# 이미지 명과 설명 mapping
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# 구두점, 숫자 제거
clean_descriptions(descriptions)
# 어휘 분석
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# 파일 저장
save_descriptions(descriptions, 'descriptions.txt')
