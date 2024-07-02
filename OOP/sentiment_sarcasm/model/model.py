import tensorflow as tf 
from model.layers.lstm_cell import LSTM 

class ProtonXRNN(tf.keras.Model):
  def __init__(self, units, embedding_size, vocab_size, input_length):
    super(ProtonXRNN, self).__init__()
    self.input_length = input_length
    self.units = units

    # Embedding để chuyển từ thành vector
    self.embedding = tf.keras.layers.Embedding(
      vocab_size,
      embedding_size,
      input_length = input_length
    )

    # Sử dụng cell LSTM đã lập trình bên trên
    self.lstm = LSTM(units, embedding_size)

    # Sau đó đưa lịch sử h của LSTM qua mạng nơ ron đơn giản
    self.classfication_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(units,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


  def call(self, sentence):
    """
    Parameters:
    sentence:
      Dạng: Tensor
      Miêu tả: Câu
      Chiều: (batch_size, input_length)
    out:
      Dạng: Tensor
      Miêu tả: Đầu ra của mô hình dự đoán
      Chiều: (batch_size, 1)
    """

    batch_size = tf.shape(sentence)[0]

    # Khởi tạo (hidden_state và context_state)

    pre_layer = tf.stack([
      tf.zeros([batch_size, self.units]),
      tf.zeros([batch_size, self.units])
    ])

    # Đưa câu qua Embedding để lấy các vector
    # embedded_sentence: (batch_size, input_length, embedding_size) # (64, 25, 100)
    embedded_sentence = self.embedding(sentence)

    # Đưa tuần tự từng từ qua LSTM + (hidden_state và cell_state)
    # lớp phía trước để thu được (hidden_state và cell_state) hiện tại


    for i in range(self.input_length):
      # : đầu tiên: Lấy batch_size
      # i Vị trí từ
      # : cuối cùng: Lấy embedding.
      # (batch_size, embedding_size)
      word = embedded_sentence[:, i, :]
      pre_layer = self.lstm(pre_layer, word)


    h, _ = tf.unstack(pre_layer)

    # Sử dụng hidden_state cuối cùng cho việc dự đoán
    return self.classfication_model(h)



