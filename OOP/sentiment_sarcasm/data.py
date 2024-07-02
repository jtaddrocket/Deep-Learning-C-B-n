import requests #thư viện gọi đường dẫn để tải về
import json
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset():
  def __init__(self):
    self.dataPath = None
    self.dataset = []
    self.labels_dataset = []
    self.tokenizer = None

  def download(self, url):
    try:
      respone = requests.get(url)
      respone.raise_for_status()
      
      self.dataPath = './sarcasm.json'
      
      # Khi tải về sẽ chia gói thành nhiều phần nhỏ sau đó sẽ nối lại
      with open(self.dataPath, 'wb') as f:
        for chunk in respone.iter_content(chunk_size=1024): #Chia thành các chunk 1 mb
          if chunk:
            f.write(chunk)

      print("Data downloaded!")

    except requests.exceptions.RequestException as e:
      print("Error downloading data: ", e)
      return

  def load_data(self):
    try:
      with open(self.dataPath, 'r') as f:
        data = json.load(f)

      for item in data:
        self.dataset.append(item['headline'])
        self.labels_dataset.append(item['is_sarcastic'])

      self.dataset = np.array(self.dataset)
      self.labels_dataset = np.array(self.labels_dataset)

      #return self.dataset, self.labels_dataset

    except FileNotFoundError:
      print("Data file not found!")
      return

  def split_data(self, train_size):
    size = int(len(self.dataset) * train_size)

    train_sentences = self.dataset[:size]
    test_sentences = self.dataset[size:]

    train_labels = self.labels_dataset[:size]
    test_labels = self.labels_dataset[size:]

    return (train_sentences, train_labels), (test_sentences, test_labels)
  
  def build_tokenizer(self, train_sentences, max_words):
    self.tokenizer = Tokenizer(num_words=max_words, oov_token="00V")
    self.tokenizer.fit_on_texts(train_sentences)

    return self.tokenizer
  
  def tokenize(self, sentences, max_length=25, truncating="post", padding="post"):
    sequences = self.tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding, truncating=truncating)
    
    return padded_sequences