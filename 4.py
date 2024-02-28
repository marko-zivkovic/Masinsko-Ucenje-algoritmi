#!pip install nltk
import nltk
import io
import pandas as pd
nltk.download()

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['disaster-tweets.csv']))

from nltk import corpus
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import re
from sklearn.utils import shuffle
print('Creating the training anfd testing set...')

df = shuffle(df)
train = df.iloc[:round(len(df)*0.4)]
train = train.append(df.iloc[round(len(df)*0.6):], ignore_index=True)
test = pd.DataFrame()
test = test.append(df.iloc[round(len(df)*0.4):round(len(df)*0.6)], ignore_index=True)
i = 0

train.isnull().sum()
test.isnull().sum()

train = train.drop(columns=['location'])
test = test.drop(columns=['location'])

def clean_text(t):
  t = re.sub(r'https?://\S+', '', t) # izbaci link
  t = re.sub(r'\n',' ', t) # izbaci novi red
  t = re.sub('\s+', ' ', t).strip() # izbaci prazna polja
  t = re.sub('[^A-Za-z0-9]+', ' ', t)
  return t

print('Cleaning the train set...')
stemmer = PorterStemmer()
stop_punc = set(stopwords.words('english')).union(set(punctuation))

#za svaki tvit
for doc in train['text']:
  doc = clean_text(doc)
  #podeli na reci
  words = wordpunct_tokenize(doc)
  #prebaci sve u lowercase
  words_lower = [w.lower() for w in words]
  #ako nije u skupu 'stopwords' unija 'interpunkcija' ostaje u words_filtered
  words_filtered = [w for w in words_lower if w not in stop_punc]
  #stemuj, odnosno ostavi koren reci
  words_stemmed = [stemmer.stem(w) for w in words_filtered]
  train['text'][i] = words_stemmed
  i += 1

print('Cleaning the testing set...')
i = 0
#za svaki tvit
for doc in test['text']:
  doc = clean_text(doc)
  #podeli na reci
  words = wordpunct_tokenize(doc)
  #prebaci sve u lowercase
  words_lower = [w.lower() for w in words]
  #ako nije u skupu 'stopwords' unija 'interpunkcija' ostaje u words_filtered
  words_filtered = [w for w in words_lower if w not in stop_punc]
  #stemuj, odnosno ostavi koren reci
  words_stemmed = [stemmer.stem(w) for w in words_filtered]
  test['text'][i] = words_stemmed
  i += 1

print('Creating the vocabulary...')

vocab_set = set()
for doc in train['text']:
  for word in doc:
    vocab_set.add(word)

vocab = list(vocab_set)

print('Vocab: ', list(zip(vocab, range(len(vocab)))))

doc_count = dict()
for word in vocab:
  doc_count[word] = 0
  for doc in train['text']:
    if word in doc:
      doc_count[word] += 1

print('Number of occurances for each word in the dictionary: ')
print(doc_count)

def numocc_score(word, doc):
  return doc.count(word)

print('Creating BOW for training...')
X = np.zeros((len(train['text']), len(vocab)), dtype=np.float32)
for doc_idx in range(len(train['text'])):
  doc = train['text'][doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = numocc_score(word, doc) # ako zamenimo sa freq_score ili numocc_score dobijamo bow
    X[doc_idx][word_idx] = cnt

print('BoW (number of occurrances) training set:')
print(X)

print('Creating TestBOW set...')
test_bow = np.zeros((len(test['text']), len(vocab)), dtype=np.float32)
for doc_idx in range(len(test['text'])):
  doc = test['text'][doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = numocc_score(word, doc) # ako zamenimo sa freq_score ili numocc_score dobijamo bow
    test_bow[doc_idx][word_idx] = cnt

class MultinomialNaiveBayes:
  def __init__(self,  nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words #sve reci
    self.pseudocount = pseudocount

  def fit(self, X, Y):
    nb_examples = X.shape[0] #svi tekstovi
    
    #verovatnoca klase = P(Klasa)
    self.priors = np.bincount(Y) / nb_examples

    #broj ponavljanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += cnt #za posmatranu klasu c, rec w se ponavlja broj puta (iz X BoW modela)

    #racunamo likelihoods = P(Rec_i | Klasa), verovatnoca da ce rec i biti u datoj klasi
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount #iz formule
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount #iz formule
        self.like[c][w] = up / down

  def predict(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = self.priors[c]
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w]) 
      probs[c] = prob
    
    prediction = np.argmax(probs)
    return prediction
  

class_names = ['Positive', 'Negative'] # 0 positive, 1 negative
model = MultinomialNaiveBayes(nb_classes=2, nb_words = len(vocab), pseudocount=1) 

#Y izvlacimo iz train seta
Y = []
for i in range(len(train)):
  Y.append(train['target'][i])

print('Y: ')
print(Y)

#X gore definisan bow
print('X: ')
print(X)

print('Training...')
model.fit(X, Y)
print('Training finished')

print('Predicting the testing set values...')
predictionarray = []
for i in test_bow:
  predictionarray.append(model.predict(i))

print('Prediction array: ')
print(predictionarray)

print('Accuracy: [number correctly predicted / total number]')
poklapanja = 0
for i in range(len(test['target'])):
  if predictionarray[i] == test['target'][i]:
    poklapanja += 1

print(poklapanja/len(test['target']))


#broj ponavljanja za svaku rec u recniku = word_count (dictionary)

corpus = train.append(test, ignore_index=True)
pos_corpus = []
neg_corpus = []

print('Creating the postive corpus...')

for i in range(len(corpus)):
  if int(corpus['target'][i]) == 0: #ako je traget 0 - pozitivan, dodaj u korpus
    pos_corpus.append(corpus['text'][i])

print('Creating the postive vocab...')

pos_vocab_set = set()   #kreiranje vokabulara za pozitivne tvitove
for doc in pos_corpus:    
  for word in doc:
    pos_vocab_set.add(word)

pos_vocab = list(pos_vocab_set)

print('Creating the dictionary with words and their occurences in the pos_corpus...')
pos_count = dict()    #kreiranje recnika sa svakom recju i njenim brojem ponavljanja u poz tvitovima
for word in pos_vocab:
  pos_count[word] = 0
  for doc in pos_corpus:
    if word in doc:
      pos_count[word] += 1

pos_sorted = sorted(pos_count.items(), key=lambda x:x[1])   #sortiranje recnika po vrednostima rastuce
pos_sorted.reverse() #namestiti da bude opadajuce
print('Five most common words in positive tweets are: [word, number of occurences]')
for i in range(5):  #ispisi prvih 5
  print(pos_sorted[i])


#-------------------------------------------------------------------------------
print('Creating the negative corpus...')

for i in range(len(corpus)):
  if int(corpus['target'][i]) == 1: #ako je traget 1 - negativan, dodaj u korpus
    neg_corpus.append(corpus['text'][i])

print('Creating the negative vocab...')

neg_vocab_set = set()   #kreiranje vokabulara za negativne tvitove
for doc in neg_corpus:
  for word in doc:
    neg_vocab_set.add(word)

neg_vocab = list(neg_vocab_set)

print('Creating the dictionary with words and their occurences in the neg_corpus...')
neg_count = dict() #kreiranje recnika sa svakom recju i njenim brojem ponavljanja u neg tvitovima
for word in neg_vocab:
  neg_count[word] = 0
  for doc in neg_corpus:
    if word in doc:
      neg_count[word] += 1

neg_sorted = sorted(neg_count.items(), key=lambda x:x[1])   #sortiranje po broju ponavljanja rastuce
neg_sorted.reverse()  #opadajuce
print('Five most common words in nega ive tweets are: [word, number of occurences]')
for i in range(5):    #ispisi prvih 5
  print(neg_sorted[i])

#--------------------------------------------------------------------------------

lr = []
print('Creating dictionary with words and their LR values...')
for w in vocab:             #popuni recnike iz neg i poz tvitovima sa nulama za reci koje ne postoje u njima a postoje u velikom recniku 
  if w not in pos_count:
    pos_count[w] = 0
  if w not in neg_count:
    neg_count[w] = 0

for w in vocab:         #izracunaj lr metriku za svaku rec i dodaj u lr niz
  if neg_count[w] != 0:
    lr.append(pos_count[w]/neg_count[w])
  else:
    lr.append(0)

lr_dict = dict()            #recnik sa recima i njihovom metrikom

for i in range(len(vocab)):
  lr_dict[vocab[i]] = lr[i]

print(lr_dict)
lr_dict_clean = dict()

for k,v in lr_dict.items():     #izbaci reci sa 0 metrikom i koje se pojavljuju u poz i neg tvitovima manje od 10 puta
  if v != 0 and pos_count[k] > 10 and neg_count[k] > 10:
    lr_dict_clean[k] = v

lr_top5 = []
lr_bottom5 = []

lr_sorted = sorted(lr_dict_clean.items(), key=lambda x:x[1])    #sortiraj i uzmi prvih 5 i poslednjih 5 reci iz ovog niza
for i in range(5):
    lr_bottom5.append(lr_sorted[i])

print('Words with lowest LR values')
for i in  lr_bottom5:
  print(i)

lr_sorted.reverse()
for i in range(5):
    lr_top5.append(lr_sorted[i])

print('Words with highest LR values')
for i in lr_top5:
  print(i)


# preciznost je uvek izmedju 70 i 80%
# 5 reci sa najmanjom lr metrikom su: kill, train, report, latest, teror
# 5 reci sa najvecom lr metrikom su: obliter, love, scream, let, fuck
# ovo znaci da se prvih 5 reci pojavljuju najvise od svih reci u negativnim tvitovima, u odnosu na pozitivne,
# dok se drugih 5 pojavljuju najvise u pozitivnim u odnosu na negativne tvitove

#LR je niz "LR" vrednosti koji za svaku rec koja se pojavljuje vise od 10 puta u pozitivnim i u negativnim 
#tvitovima. Za svaku rec se uzima broj ponavljanja u pozitivnim tvitovima i deli se sa brojem ponavljanja u negativnim tvitovima.

