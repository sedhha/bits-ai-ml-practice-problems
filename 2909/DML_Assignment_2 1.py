import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import time
import uuid
import logging
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from threading import Thread
import networkx as nx
import yfinance as yf
import signal
import sys
import traceback
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def signal_handler(sig, frame):
    logging.info('Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train_model(ticker='AAPL', start_date='2010-01-01', end_date='2025-01-01'):
    stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d')
    
    # Normalize the data using min-max scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    features = scaler.fit_transform(features)  # Normalize features
    targets = stock_data['Close'].shift(-1).dropna().values
    targets = targets[:-1]  # Match features length
    
    model = Sequential([
        Dense(64, input_dim=5, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(features, targets, epochs=100, batch_size=32)
    model.save_weights('model.weights.h5')
    logging.info("Model trained and weights saved as model.weights.h5")

# Producer Implementation
class DataProducer:
    def __init__(self, topic, bootstrap_servers='localhost:9092', interval=1, ticker='AAPL'):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.topic = topic
        self.interval = interval
        self.ticker = ticker
        self.data_count = 0
        self.timestamps = []

    def generate_data(self):
        while True:
            try:
                random_date = np.random.choice(pd.date_range(start=(datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d')))
                random_date = pd.Timestamp(random_date).to_pydatetime()
                stock_data = yf.Ticker(self.ticker).history(start=random_date, end=random_date + pd.Timedelta(days=1), interval='1m')
                if stock_data.empty:
                    continue
                latest_data = stock_data.iloc[-1]
                data = {
                    'id': str(uuid.uuid4()),
                    'features': [latest_data['Open'], latest_data['High'], latest_data['Low'], latest_data['Close'], latest_data['Volume']],
                    'timestamp': datetime.now().isoformat()
                }
                self.producer.send(self.topic, data)
                logging.info(f"Produced data: {data}")
                self.data_count += 1
                self.timestamps.append(time.time())
                time.sleep(self.interval)
            except Exception as e:
                logging.error(f"Error in producing data: {e}")
                traceback.print_exc()

# Consumer Implementation with Neural Network Prediction
class DataConsumer:
    def __init__(self, input_topic, output_topic, bootstrap_servers='localhost:9092'):
        self.consumer = KafkaConsumer(input_topic, bootstrap_servers=bootstrap_servers, value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.output_topic = output_topic
        self.model = self.load_model()

    def load_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=5, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.load_weights('model.weights.h5')
        return model

    def preprocess_data(self, data):
        scaler = MinMaxScaler()
        
        features = np.array(data['features']).reshape(1, -1)
        features = scaler.fit_transform(features)  # Normalize input data
        return features

    def predict(self):
        for message in self.consumer:
            try:
                data = message.value
                features = self.preprocess_data(data)
                prediction = float(self.model.predict(features)[0][0])
                output_data = {
                    'id': data['id'],
                    'input_features': data['features'],
                    'prediction': prediction,
                    'prediction_timestamp': datetime.now().isoformat()
                }
                self.producer.send(self.output_topic, output_data)
                logging.info(f"Produced prediction: {output_data}")
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                traceback.print_exc()

# Testing and Evaluation
class Monitor:
    def __init__(self, topic, bootstrap_servers='localhost:9092'):
        self.consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        self.message_count = 0
        self.total_latency = 0
        self.total_squared_error = 0

    def monitor(self):
        for message in self.consumer:
            try:
                data = message.value
                self.message_count += 1
                latency = (datetime.now() - datetime.fromisoformat(data['prediction_timestamp'])).total_seconds()
                self.total_latency += latency
                avg_latency = self.total_latency / self.message_count

                # Assuming the true value is part of the data for evaluation
                true_value = data['input_features'][3]
                prediction = data['prediction']
                squared_error = (true_value - prediction) ** 2
                self.total_squared_error += squared_error
                mean_squared_error = self.total_squared_error / self.message_count

                logging.info(f"Message Count: {self.message_count}, Latency: {latency}, Average Latency: {avg_latency}, Mean Squared Error: {mean_squared_error}, Squared Error: {squared_error}")

                plt.clf()
                plt.subplot(4, 1, 1)
                plt.plot(self.message_count, latency, 'go-', label=f'Latency: {latency:.4f}s')
                plt.xlabel('Message Count')
                plt.ylabel('Latency (s)')
                plt.legend()

                plt.subplot(4, 1, 2)
                plt.plot(self.message_count, squared_error, 'mo-', label=f'Mean Squared Error(Iteration): {squared_error:.4f}')
                plt.xlabel('Message Count')
                plt.ylabel('Mean Squared Error')
                plt.legend()

                plt.subplot(4, 1, 3)
                plt.plot(self.message_count, avg_latency, 'bo-', label=f'Average Latency: {avg_latency:.4f}s')
                plt.xlabel('Message Count')
                plt.ylabel('Average Latency (s)')
                plt.legend()

                plt.subplot(4, 1, 4)
                plt.plot(self.message_count, mean_squared_error, 'ro-', label=f'Mean Squared Error: {mean_squared_error:.4f}')
                plt.xlabel('Message Count')
                plt.ylabel('Mean Squared Error')
                plt.legend()

                plt.pause(0.01)

            except Exception as e:
                logging.error(f"Error in monitoring: {e}")
                traceback.print_exc()
    


if __name__ == "__main__":
    if not os.path.exists('model.weights.h5'):
        train_model()
    
    producer = DataProducer(topic='input-data')
    consumer = DataConsumer(input_topic='input-data', output_topic='predictions')
    monitor = Monitor(topic='predictions')

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(producer.generate_data)
        executor.submit(consumer.predict)
    
    # Run monitor in main thread
    monitor.monitor()

