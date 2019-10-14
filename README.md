![image](inpredo_logo.png)

Inpredo (INtelligent PREDictions) is an AI which literally looks into financial charts and predicts stock movements.

### First Step - Create Training Data:

Before start to train a Convolutional Neural Network, first you need to create a
training dataset. As a starting point you can use one of the following timeseries financial data:

- BTC-USD Hourly price data; btcusd-1h.csv
- EUR-USD Hourly price data; eurusd.csv
- Gold-USD Hourly price data; xausd.csv

Since I am too lazy to automate all this, you need to enter your CSV file into the following line:
For example, if you wanna train your AI on euro dollar prices:

`ad = genfromtxt('/financial_data/eurusd.csv', delimiter=',' ,dtype=str)`

This code line is found under `graphwerk.py` which is the factory that produces images out of time series financial data.
After running `graphwerk.py` it will take some time to write single jpg files under data/train folder.
When script is done writing, then you need to take randomly roughly 20 percent of the training data and put it into validation data.
You need this to be able to train a neural network and yes, I was too lazy to automate that as well.

## Second Step - Train the AI!

Now we have the training and validation datasets in place, you can start training the AI model.
For this you just need to run `train-binary.py` and this script will start using the dataset make a AI model out of it.
When the model training is complete, it will generate a model and weights file under the models directory.

## Third Step - Load Models and Predict

You can run predictions using `predict-binary.py` script. Use the `predict(file)`
and use the path of the jpg file you want to predict. Result of the script will be a buy, sell or not confident message.

## Last words

Actually this project is much bigger, but for some reasons I only put the training and data generation part here.
There is another part of the project which actually trades in real time using nothing but AI Models from this project.

For people who wants to go experimental, don't forget that you can lose money in real markets and I am not accountable for your stupitidy if you choose to use this project to trade with your own money.

Medium article for in depth explanation of the project: https://medium.com/@cderinbogaz/making-a-i-that-looks-into-trade-charts-62e7d51edcba