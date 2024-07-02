import tensorflow as tf 

class LSTM(tf.keras.layers.Layer):
  def __init__(self, units, inp_shape):
    super(LSTM, self).__init__()
    self.units = units
    self.inp_shape = inp_shape
    self.W = self.add_weight("W", shape=(4, self.units, self.inp_shape))
    self.U = self.add_weight("U", shape=(4, self.units, self.units))


  def call(self, pre_layer, x):
    pre_h, pre_c = tf.unstack(pre_layer)

    # Cổng kiểm soát đầu vào: Input Gate

    i_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul( pre_h, tf.transpose(self.U[0])))

    # Cổng kiểm soát số lượng dữ liệu giữ lại/quên đi: Forget Gate
    f_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul( pre_h, tf.transpose(self.U[1])))

    # Cổng kiểm soát dữ liệu đầu ra: Output Gate
    o_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul( pre_h, tf.transpose(self.U[2])))

    # Đây giống SimpleRNN và được coi là thông tin mới (bộ nhớ mới)

    n_c_t = tf.nn.tanh(tf.matmul(x, tf.transpose(self.W[3])) + tf.matmul( pre_h, tf.transpose(self.U[3])))

    # Kết hợp việc giữ lại thông tin + bổ sung thêm thông tin mới

    c = tf.multiply(f_t, pre_c) + tf.multiply(i_t, n_c_t)

    # Cho phép bao nhiêu thông tin thoát khỏi cell

    h = tf.multiply(o_t, tf.nn.tanh(c))

    return tf.stack([h, c])