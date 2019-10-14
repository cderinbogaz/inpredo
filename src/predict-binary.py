import os
import numpy as np
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


img_width, img_height = 150, 150
model_path = '../src/models/model.h5'
weights_path = '../src/models/weights'
model = load_model(model_path)
test_path = '../data/validation'
csv_path = '../src/models/results.csv'


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    if result[0] > 0.9:
      print("Predicted answer: Buy")
      answer = 'buy'
      print(result)
      print(array)
    else:
      print("Predicted answer: Not confident")
      answer = 'n/a'
      print(result)
    return answer
  else:
    if result[1] > 0.9:
        print("Predicted answer: Sell")
        answer = 'sell'
        print(result)
    else:
        print("Predicted answer: Not confident")
        answer = 'n/a'
        print(result)
    return answer


def predict_files(test_path):
  truebuy = 0
  falsebuy = 0
  truesell = 0
  falsesell = 0
  na = 0
  for i, ret in enumerate(os.walk(str(test_path) + '/buy')):
    for i, filename in enumerate(ret[2]):
      if filename.startswith("."):
        continue
      print("Label: buy")
      result = predict(ret[0] + '/' + filename)
      if result == "buy":
        truebuy += 1
      elif result == 'n/a':
        print('no action')
        na +=1
      elif result == 'sell':
        falsebuy += 1


  for i, ret in enumerate(os.walk(str(test_path) + '/sell')):
    for i, filename in enumerate(ret[2]):
      if filename.startswith("."):
        continue
      print("Label: sell")
      result = predict(ret[0] + '/' + filename)
      if result == "sell":
        truesell += 1
      elif result == 'n/a':
        print('no action')
        na += 1
      elif result == 'buy':
        falsesell += 1

  print("True buy: ", truebuy)
  print("True sell: ", truesell)
  print("False buy: ", falsebuy)  # important
  print("False sell: ", falsesell)
  print("no action:", na)
  precision = truesell / (truesell + falsesell)
  precision2 = truebuy / (truebuy + falsebuy)
  recall = truebuy / (truebuy + falsesell)
  print("Sell Precision: ", precision)
  print("Buy Precision", precision2)
  print("Recall: ", recall)
  precision1 = (truesell + truebuy) / (truesell + truebuy + falsesell + falsebuy)
  print("Precision: ", precision1)
  f_measure = (2 * recall * precision) / (recall + precision)
  print("F-measure: ", f_measure)
  return precision1, na
