0. 딥러닝 핵심 Task 실습 과정 안내
1. AI 기본 훑어보기
  1-1 AI, ML, DL 개념
    AI > ML > DL
    인공 뉴런과 인공 신경망
      FNN (FCN): Feedforward Neural Network 또는 Fully-connected Neural Network
        FNN: 노드끼리 마구잡이로 데이터를 앞으로 보냈다, 뒤로 보냈다 하는 등... 절대 순서가 꼬이지 않고 한쪽 방향으로만의 흐름만 존재
        FCN: 하나의 노드는 다음 레이어의 모든 노드와 연결된다고 하여, 완전히 연결되었다
      CNN : Convolutional Neural Network
        인공신경망은 Image 형태와 같이 가로x세로의 공간정보를 가진 데이터를 다루는 데 특화된 신경망 종류
        Convolution: 입력텐서로부터 어떤 특징이 있는지를 훑어내고,
        Pooling: convolution이 추출한 특징으로부터 대표적인 것들만을 추려내는 subsampling 작업
      RNN : Recurrent Neural Network
        인공신경망은 순차적인 흐름을 갖는 데이터를 다루는 데 특화된 신경망 종류
        매 timestep마다 데이터가 들어오는데, 입력된 데이터 뿐 아니라 과거의 처리 이력을 반영하여 출력하는 방식
  1-2 실습환경셋팅

2. 비정형 데이터 처리
  2-1 Data Normalization & Augmentation
    Normalization
      데이터 값들의 범위를 0~1 사이의 값으로 축소
      딥러닝 모델의 학습 속도를 빠르게 하기 위해 + 데이터값 크기에 따른 영향 분산
    Data standardization
      사용자가 데이터를 처리하고 분석 할 수 있도록 데이터를 공통 형식으로 변환
    Augmentation
      인위적으로 데이터를 늘려서 과적합(overfitting)을 방지하고 일반화 성능을 높임
      예) 영상
      일정 영역 noise 채우기, 두 이미지 섞기, 크기&배율 조정 및 회전
      example: image augmentation 원본 펼치기
    (실습파일) 2-1.실습_데이터_전처리.ipynb
  2-2 Embedding
    연산이 어려운 범주형(categorical)이나 텍스트 데이터를 계산 가능한 벡터로 변환 하는 과정
    1) One-hot encoding (원-핫 인코딩)
      (정의) 모든 데이터의 범주를 쭈루룩 줄세워 사전을 만들고, 순서대로 번호를 붙이는 작업
      (과정)
        내가 가진 모든 범주의 중복을 제거한 사전(vocab) 생성
        사전(vocab)의 길이(중복 제외한 범주 수)만큼 벡터를 정의하고,
         벡터에서 토큰의 순서에 해당하는 위치만 1, 나머지는 0의 값으로 채워 벡터를 구성
        가진 데이터의 범주들을 해당 벡터로 교체
      (효과) 원-핫 인코딩 방식으로 데이터를 임베딩하게되면 모든 범주간 √2만큼의 서로 동일한 거리를 갖게 되어, 여러 범주를 동등하게 취급
      (단점) 범주가 다양하고 수가 많을수록 범주 하나를 표현하기 위해서 굉장히 길이가 긴 벡터를 필요
    2) Embedding
      데이터를 다차원의 벡터 공간의 한 점으로 매핑
      (실습파일) 2-2.실습_Embedding.ipynb
      example_Embedding 원본 펼치기
      Tokenizing (Parsing)
        한 덩이로 되어있는 문장을 인공신경망에 인식시키기 위해서, 임베딩 가능한 단위로 쪼개는 작업
        토큰(token): 쪼개진 단위
        한국어의 경우엔 교착어라는 언어 특성상 문장을 형태소 단위로 자르거나 음절단위를 사용
      사전 만들기
        음절 단위로 텍스트를 쪼개기로 했다면, 내가 처리할 모든 음절의 종류를 모아 사전(vocab)을 구축
        특수 Token 추가
          [PAD]: Padding → 미니배치간 데이터 길이를 맞추기 위해 빈 부분을 채우기 위해 사용
          [OOV]: Out of Vocabulary → vocab 사전에 없는 처음보는 글자
      input 변환
      Embedding 및 모델 학습
        밑에 3-1에서 실습

3. 딥러닝 핵심 TASK
  3-1 기본 모델 활용 (FNN/CNN/RNN)
    FNN
      MNIST 흑백 손글씨
      (실습파일) 3-1.실습_1_FNN.ipynb
        Functional API 방식으로 구현됨
    CNN
      개/고양이 영상 분류
      (실습파일) 3-1.실습_2_CNN.ipynb
    RNN
      영화 리뷰 긍정/부정 분류하는 텍스트 감성분류: RNN(LSTM)
      (실습파일) 3-1.실습_3_RNN.ipynb
  3-2 Transfer Learning
    한번 만들어진 딥러닝 모델을 재활용하여 쓸 수 있는 기법
    모델이 사전에 학습한 지식은 잘 유지하면서도, 새로운 태스크를 수행하는 데에 필요한 지식을 추가로 습득할 수 있도록 함
    (실습파일) 3-2.실습_Transfer_Learning.ipynb
      (1차 시험 내용??)
  3-3 Multi-label Classification
    one-hot 인코딩 양식으로 라벨이 구성되어있었다면, Multi-label classification의 경우에는 해당하는 클래스의 인덱스가 전부 1로 구성
      예) 영화 장르
      가정 : [액션, 코미디, 로맨스, 공포, 드라마]의 다섯 클래스
      영화A : 로맨스이면서도 코미디인 영화 → 이 영화의 라벨은 [0,1,1,0,0]
    (실습파일) 3-3.실습_Multilabel_Classification.ipynb
      (1차 시험 내용??)
  3-4 Metric Learning
    객체 사이의 유사도를 파악
      같다/다르다라는 유사도를 나타내기 위해서는 거리(distance)가 필요
      거리가 가까우면 같은 종류의 사물이고, 멀면 다른 종류의 사물이라고 할 수 있음
    Metric Learning의 구성은 크게 두 가지
      (1) 어떤 거리(유사도)를 사용할 것인가
        Euclidean distance, Cosine similarity, Wasserstein distance 등
      (2) 유사도 학습을 위해 어떤 loss 함수로 학습할 것인가
        Triplet loss: tf.einsum_설명_v1.pptx
      (실습파일) 3-4.실습_Metric_Learning.ipynb
