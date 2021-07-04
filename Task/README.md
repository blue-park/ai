0. ������ �ٽ� Task �ǽ� ���� �ȳ�
1. AI �⺻ �Ⱦ��
  1-1 AI, ML, DL ����
    AI > ML > DL
    �ΰ� ������ �ΰ� �Ű��
      FNN (FCN): Feedforward Neural Network �Ǵ� Fully-connected Neural Network
        FNN: ��峢�� �������̷� �����͸� ������ ���´�, �ڷ� ���´� �ϴ� ��... ���� ������ ������ �ʰ� ���� �������θ��� �帧�� ����
        FCN: �ϳ��� ���� ���� ���̾��� ��� ���� ����ȴٰ� �Ͽ�, ������ ����Ǿ���
      CNN : Convolutional Neural Network
        �ΰ��Ű���� Image ���¿� ���� ����x������ ���������� ���� �����͸� �ٷ�� �� Ưȭ�� �Ű�� ����
        Convolution: �Է��ټ��κ��� � Ư¡�� �ִ����� �Ⱦ��,
        Pooling: convolution�� ������ Ư¡���κ��� ��ǥ���� �͵鸸�� �߷����� subsampling �۾�
      RNN : Recurrent Neural Network
        �ΰ��Ű���� �������� �帧�� ���� �����͸� �ٷ�� �� Ưȭ�� �Ű�� ����
        �� timestep���� �����Ͱ� �����µ�, �Էµ� ������ �� �ƴ϶� ������ ó�� �̷��� �ݿ��Ͽ� ����ϴ� ���
  1-2 �ǽ�ȯ�����

2. ������ ������ ó��
  2-1 Data Normalization & Augmentation
    Normalization
      ������ ������ ������ 0~1 ������ ������ ���
      ������ ���� �н� �ӵ��� ������ �ϱ� ���� + �����Ͱ� ũ�⿡ ���� ���� �л�
    Data standardization
      ����ڰ� �����͸� ó���ϰ� �м� �� �� �ֵ��� �����͸� ���� �������� ��ȯ
    Augmentation
      ���������� �����͸� �÷��� ������(overfitting)�� �����ϰ� �Ϲ�ȭ ������ ����
      ��) ����
      ���� ���� noise ä���, �� �̹��� ����, ũ��&���� ���� �� ȸ��
      example: image augmentation ���� ��ġ��
    (�ǽ�����) 2-1.�ǽ�_������_��ó��.ipynb
  2-2 Embedding
    ������ ����� ������(categorical)�̳� �ؽ�Ʈ �����͸� ��� ������ ���ͷ� ��ȯ �ϴ� ����
    1) One-hot encoding (��-�� ���ڵ�)
      (����) ��� �������� ���ָ� �޷�� �ټ��� ������ �����, ������� ��ȣ�� ���̴� �۾�
      (����) 
        ���� ���� ��� ������ �ߺ��� ������ ����(vocab) ����
        ����(vocab)�� ����(�ߺ� ������ ���� ��)��ŭ ���͸� �����ϰ�,
         ���Ϳ��� ��ū�� ������ �ش��ϴ� ��ġ�� 1, �������� 0�� ������ ä�� ���͸� ����
        ���� �������� ���ֵ��� �ش� ���ͷ� ��ü
      (ȿ��) ��-�� ���ڵ� ������� �����͸� �Ӻ����ϰԵǸ� ��� ���ְ� ��2��ŭ�� ���� ������ �Ÿ��� ���� �Ǿ�, ���� ���ָ� �����ϰ� ���
      (����) ���ְ� �پ��ϰ� ���� �������� ���� �ϳ��� ǥ���ϱ� ���ؼ� ������ ���̰� �� ���͸� �ʿ�
    2) Embedding
      �����͸� �������� ���� ������ �� ������ ����
      (�ǽ�����) 2-2.�ǽ�_Embedding.ipynb
      example_Embedding ���� ��ġ��
      Tokenizing (Parsing)
        �� ���̷� �Ǿ��ִ� ������ �ΰ��Ű���� �νĽ�Ű�� ���ؼ�, �Ӻ��� ������ ������ �ɰ��� �۾�
        ��ū(token): �ɰ��� ����
        �ѱ����� ��쿣 �������� ��� Ư���� ������ ���¼� ������ �ڸ��ų� ���������� ���
      ���� �����
        ���� ������ �ؽ�Ʈ�� �ɰ���� �ߴٸ�, ���� ó���� ��� ������ ������ ��� ����(vocab)�� ����
        Ư�� Token �߰�
          [PAD]: Padding �� �̴Ϲ�ġ�� ������ ���̸� ���߱� ���� �� �κ��� ä��� ���� ���
          [OOV]: Out of Vocabulary �� vocab ������ ���� ó������ ����
      input ��ȯ
      Embedding �� �� �н�
        �ؿ� 3-1���� �ǽ�

3. ������ �ٽ� TASK
  3-1 �⺻ �� Ȱ�� (FNN/CNN/RNN)
    FNN
      MNIST ��� �ձ۾�
      (�ǽ�����) 3-1.�ǽ�_1_FNN.ipynb
        Functional API ������� ������
    CNN
      ��/������ ���� �з�
      (�ǽ�����) 3-1.�ǽ�_2_CNN.ipynb
    RNN
      ��ȭ ���� ����/���� �з��ϴ� �ؽ�Ʈ �����з�: RNN(LSTM)
      (�ǽ�����) 3-1.�ǽ�_3_RNN.ipynb
  3-2 Transfer Learning
    �ѹ� ������� ������ ���� ��Ȱ���Ͽ� �� �� �ִ� ���
    ���� ������ �н��� ������ �� �����ϸ鼭��, ���ο� �½�ũ�� �����ϴ� ���� �ʿ��� ������ �߰��� ������ �� �ֵ��� ��
    (�ǽ�����) 3-2.�ǽ�_Transfer_Learning.ipynb
      (1�� ���� ����??)
  3-3 Multi-label Classification
    one-hot ���ڵ� ������� ���� �����Ǿ��־��ٸ�, Multi-label classification�� ��쿡�� �ش��ϴ� Ŭ������ �ε����� ���� 1�� ����
      ��) ��ȭ �帣
      ���� : [�׼�, �ڹ̵�, �θǽ�, ����, ���]�� �ټ� Ŭ����
      ��ȭA : �θǽ��̸鼭�� �ڹ̵��� ��ȭ �� �� ��ȭ�� ���� [0,1,1,0,0]
    (�ǽ�����) 3-3.�ǽ�_Multilabel_Classification.ipynb
      (1�� ���� ����??)
  3-4 Metric Learning
    ��ü ������ ���絵�� �ľ�
      ����/�ٸ��ٶ�� ���絵�� ��Ÿ���� ���ؼ��� �Ÿ�(distance)�� �ʿ�
      �Ÿ��� ������ ���� ������ �繰�̰�, �ָ� �ٸ� ������ �繰�̶�� �� �� ����
    Metric Learning�� ������ ũ�� �� ����
      (1) � �Ÿ�(���絵)�� ����� ���ΰ�
        Euclidean distance, Cosine similarity, Wasserstein distance ��
      (2) ���絵 �н��� ���� � loss �Լ��� �н��� ���ΰ�
        Triplet loss: tf.einsum_����_v1.pptx
      (�ǽ�����) 3-4.�ǽ�_Metric_Learning.ipynb