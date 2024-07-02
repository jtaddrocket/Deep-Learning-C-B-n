from data import Dataset
from model import ProtonXRNN
import tensorflow as tf
from argparse import ArgumentParser

# Giúp người clone có thể thay đổi các chỉ số học tập
parser = ArgumentParser()
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning-rate", default=1e-6, type=int)
parser.add_argument("--max-length", default=25, type=int)
parser.add_argument("--units", default=128, type=int)
parser.add_argument("--embedding-size", default=100, type=int)

args = parser.parse_args()

print("Step 1: Loading data...")

dataset = Dataset()

dataset.download(url='https://storage.googleapis.com/learning-datasets/sarcasm.json')

dataset.load_data()

(train_sentences, train_labels), (test_sentences, test_labels) = dataset.split_data(train_size=0.8)

tokenizer = dataset.build_tokenizer(train_sentences, max_words=10000)

tokenized_train, tokenized_test = dataset.tokenize(train_sentences), dataset.tokenize(test_sentences) 

print("Step 2: Training...")

units = args.units
embedding_size = args.embedding_size
max_length = args.max_length
vocab_size = len(tokenizer.index_word) + 1
input_length = max_length

# Khởi tạo đối tượng protonxrnn
protonxrnn = ProtonXRNN(units, embedding_size, vocab_size, input_length)


protonxrnn.compile(
    tf.keras.optimizers.Adam(args.learning_rate) , loss='binary_crossentropy', metrics=['acc']
)

# Tiến hành training
protonxrnn.fit(tokenized_train, train_labels, validation_data=(tokenized_test, test_labels) ,batch_size=args.batch_size, epochs=args.epochs)