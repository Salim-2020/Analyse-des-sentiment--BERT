from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn import metrics
from transformers import CamembertTokenizer
import numpy as np
import string
desired_width=320


pd.set_option('display.width', desired_width)

pd.set_option('display.max_columns',10)
data = pd.read_csv('./avis/nouveau/Carrefour.csv')
#print(data)
import re
from nltk.corpus import stopwords
data = data.reset_index(drop=True)
STOPWORDS = set(stopwords.words('french'))
def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(r"\d", "", text) #remove number
    text = re.sub(r"[^\w\d'\s]+", '', text)
    text = re.sub(r"\s+", " ", text, flags=re.I) # remove space
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text
data['Texte'] = data['Texte'].apply(clean_text)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
data["Texte"] = data['Texte'].apply(remove_punctuations)


data['phrase_longeur'] = [len(t) for t in data.Texte]
print(data)

X=data.loc[:, 'Texte'].values
y=data.loc[:, 'Label'].values

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 600
# This is fixed.
EMBEDDING_DIM = 100
# --------pas de sequence----------------#
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
reviews_len = [len(tokenizer.encode(review, max_length=320))
                          for review in X]
print("Average length: {:.1f}".format(np.mean(reviews_len)))
print("Max length: {}".format(max(reviews_len)))

vocab = tokenizer.vocab_size
print(vocab)
import numpy as np

def encode_reviews(tokenizer, reviews, max_length):
    token_ids = np.zeros(shape=(len(reviews), max_length),
                         dtype=np.int32)
    for i, review in enumerate(reviews):
        encoded = tokenizer.encode(review, max_length=max_length)
        token_ids[i, 0:len(encoded)] = encoded
    return  token_ids

encoded_X = encode_reviews(tokenizer, X,320)
print(encoded_X.shape)





import numpy as np
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import StandardScaler


# deviser data
X_train, X_test, Y_train, Y_test = train_test_split(encoded_X,y, test_size = 0.2, shuffle=True, random_state=42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(X_train)

#--------------------------------------------------------------------------------------------------#
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras.layers import LSTM,Embedding,MaxPooling1D,GRU,Flatten,Bidirectional
from keras.layers import Dropout,Conv1D,SpatialDropout1D,MaxPooling1D,GlobalMaxPooling1D
embedding_vector_length = 64
from keras.optimizers import Adam,SGD,RMSprop
opt1  = Adam(lr=4e-5)
opt2 = SGD(lr=0.0001)
opt3 = RMSprop(lr=0.001)
sgd = SGD(lr=0.0004)
import tensorflow as tf
optimizer = tf.keras.optimizers.RMSprop (0.0099)
from keras.regularizers import l1
from keras import regularizers
import time
debut = time.time()
regressor = Sequential()
regressor.add(Embedding(input_dim= vocab , output_dim = embedding_vector_length, input_length=encoded_X.shape[1]))
regressor.add(SpatialDropout1D(0.1))
regressor.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
regressor.add(GlobalMaxPooling1D())
regressor.add(Dropout(0.1))
regressor.add(Dense(80))
regressor.add(Dropout(0.1))
regressor.add(Dense(1,activation='relu'))
from keras.utils.vis_utils import plot_model

plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
regressor.compile(loss='mae', optimizer=opt1,metrics=['mae', 'mse'])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=4, verbose=2, mode='min',
                        )
history =  regressor.fit(X_train,Y_train,validation_data=(X_test,Y_test),verbose=2,epochs=50)
#print(regressor.summary())
fin = time.time()
print(debut, fin)
print("temps de calcul", fin - debut)
print('\n')
#_____________________MESURE ERROR MSE = 1/n [ somme(Y-Y_pred)^2]__________________________________#
# Predict
pred_test = regressor.predict(X_test)
pred_train= regressor.predict(X_train)
# Measure MSE_test error.
score1 = metrics.mean_squared_error(pred_test,Y_test)
score2 =metrics.mean_squared_error(pred_train,Y_train)
score3= sqrt(score2)
print("Erreur quadratique moyenne (MSE_X_test): {}".format(score1))
print("Erreur relatuve moyenne (MAE_X_train): {}".format(score2))
print(" (RMSE_X_train): {}".format(score3))
import numpy as np
# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred_test,Y_test))
print("racine Erreur quadratique moyenne (RMSE_ X_test): {}".format(score))
# mesure error MSE_train
print("racine Erreur quadratique moyenne (RMSE_ X_train)", np.sqrt(mean_squared_error(Y_train,pred_train)))
print('\n')
#-------------------------------------------------------------------#
train_mse = regressor.evaluate(X_train, Y_train, verbose=0)
test_mse = regressor.evaluate(X_test, Y_test, verbose=0)
print(train_mse)
print(test_mse)
# plot loss during training
from matplotlib import pyplot
pyplot.title('Loss / val_loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# ------SAVE BEST MODEL---------------------------------#
regressor.save('tweet_sentiment_extraction.h5')
from keras.models import load_model
model = load_model('tweet_sentiment_extraction.h5')
para = model.evaluate(X_test, Y_test)
print(para)
print()
print("val_loss :", para[0], 'loss :', para[1])
#------------------------------------------------------------------#
#-----------------PREDECTION 10 EXP--------------------------------#
example_batch_train = X_train
example_batch_test = X_test

print("predection taille", example_batch_train.shape, example_batch_test.shape)
example_result_train = regressor.predict(example_batch_train)
example_batch_test=regressor.predict(example_batch_test)
print(example_result_train)
print(example_batch_test)
#-----------------------------------------------------------------#
#_________________________VISUALISATION_________________________________________#
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#---------------------------------------------------------------------#
# TRACER PREDECTION.
import matplotlib.pyplot as plt
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
# Plot the chart
chart_regression(pred_test.flatten() ,Y_test)
#-----------------------------------------------------------------------#
total = len(Y_test)

good = 0
top1 = 0

for i in range(len(pred_test)):
    if np.round(pred_test[i]) == Y_test[i] or np.round(pred_test[i] + 0.5) == Y_test[i] or np.round(
            pred_test[i] - 0.5) == Y_test[i]:
        good += 1
    if np.round(pred_test[i]) == Y_test[i]:
        top1 += 1

print("presecion test", good / total)
print("TOP 1 : ", top1 / total)
#----------------------------------------_#
total = len(Y_train)

good = 0
top1 = 0

for i in range(len(pred_train)):
    if np.round(pred_train[i]) == Y_train[i] or np.round(pred_train[i] + 0.5) == Y_train[i] or np.round(
            pred_train[i] - 0.5) == Y_train[i]:
        good += 1
    if np.round(pred_train[i]) == Y_train[i]:
        top1 += 1

print("presecion train", good / total)
print("TOP 1 : ", top1 / total)