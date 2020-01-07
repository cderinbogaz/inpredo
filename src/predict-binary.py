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


tb = 0
ts = 0
fb = 0
fs = 0
na = 0

for i, ret in enumerate(os.walk(data_path + '/test/buy')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: buy")
    result = predict(ret[0] + '/' + filename)
    if result == "buy":
      tb += 1
    elif result == 'n/a':
      print('no action')
      na += 1
    else:
      fb += 1

for i, ret in enumerate(os.walk(data_path + '/test/sell')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: sell")
    result = predict(ret[0] + '/' + filename)
    if result == "sell":
      ts += 1
    elif result == 'n/a':
      print('no action')
      na += 1
    else:
      fs += 1

"""
Check metrics
"""
print("True buy: ", tb)
print("True sell: ", ts)
print("False buy: ", fb)  # important
print("False sell: ", fs)
print("No action", na)

precision = (tb+ts) / (tb + ts + fb + fs)
recall = tb / (tb + fs)
print("Precision: ", precision)
print("Recall: ", recall)

f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
