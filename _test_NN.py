import asyncio
import os
import time

import aiohttp
import numpy
import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.src.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import functions
from my_config.trade_config import Config  # Файл конфигурации торгового робота


def check_nn():
    inputs = keras.Input(shape=(784,), name='quote_data')
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

    # version with Adam optimization is a stochastic gradient descent method
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary()


def check_nn2():
    model = keras.Sequential([
        Dense(256, input_dim=784, activation='tanh', bias_initializer='he_normal'),
        Dense(256, input_dim=256, activation='tanh', bias_initializer='he_normal'),
        Dense(1, input_dim=256, activation='linear', bias_initializer='he_normal')
    ])
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.summary()


def check_nn3():
    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(200, 7)),  # Input layer with shape (200, 7)
        tf.keras.layers.Flatten(),  # Flatten the input to a 1D array
        tf.keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(2)  # Output layer with 2 neurons for the 2 numbers
    ])

    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimizer='adam', loss='mse')

    # Print the model summary
    model.summary()


def check_nn4():
    # Example input data with 200 candles, each having 4 features (open, high, low, close)
    num_candles = 5
    candle_features = 4

    # Generate example input data with random values
    input_data = np.random.rand(num_candles, candle_features)
    print(input_data)
    # Flatten the input data into a single vector
    flattened_input = input_data.flatten()

    print(flattened_input)
    # Reshape the flattened_input into a 1D array with a single sample and 800 features
    reshaped_input = flattened_input.reshape(1, -1)
    print(reshaped_input)
    # Build the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=reshaped_input.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))  # Output layer with 2 neurons

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    # Example output data with 2 numbers for each input (N candles)
    output_data = np.random.rand(1, 2)  # Replace this with your actual output data

    # Train the model
    model.fit(reshaped_input, output_data, epochs=10, batch_size=32)

    # Example prediction for new input data with 200 candles
    new_input_data = np.random.rand(num_candles, candle_features)
    flattened_new_input = new_input_data.flatten()
    reshaped_new_input = flattened_new_input.reshape(1, -1)
    predictions = model.predict(reshaped_new_input)

    print("Predicted output for new input data:")
    print(predictions)


def check_nn5():
    # Load and preprocess your quote data from the exchange (assumed to be in a CSV file)
    data = pd.read_csv('exchange_data.csv')

    # Prepare the data
    X = data.drop('target_column', axis=1).values
    y = data['target_column'].values

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')


def check_data_prep():
    ticker = "SiU3"
    _tf = "M10"
    _filename = os.path.join(os.path.join("NN_futures", "csv"), f"{ticker}_{_tf}.csv")
    df = functions.load_candles(_filename)
    _mean = df["open"].mean()
    print(f"len df={len(df)} mean_open={_mean}")
    last_10 = df.tail(5)
    print("last 5")
    print(last_10)
    print("3 columns")
    res = last_10[["open", "close", "volume"]]
    print(res)
    print("3 columns values")
    res = res.values
    print(res)
    print("3 columns values flatten")
    res = res.flatten()
    print(res)
    _elem_in_row = 5
    print(f"3 columns values flatten - reshape -1 - unbounded rows, {_elem_in_row} elements in row")
    res = res.reshape(-1, _elem_in_row)
    print(res)
    print("Evaluation")
    _look_forward = 2
    print(functions.prepare_data_and_eval(df, df.last_valid_index()-_look_forward, 3, _look_forward, functions.evaluate))


def check_my_nn():
    start = time.time()
    ticker = "SiU3"
    _tf = "M1"
    _filename = os.path.join(os.path.join("NN_futures", "csv"), f"{ticker}_{_tf}.csv")
    df = functions.load_candles(_filename)

    cur_run_folder = os.path.abspath(os.getcwd())
    root_folder = Config.root_folder

    period_vwma_slow = Config.period_vwma_slow  # период медленной SMA
    period_vwma_fast = Config.period_vwma_fast  # период быстрой SMA
    period_vwma_vfast = Config.period_vwma_vfast  # период супер быстрой SMA
    look_forward = Config.look_forward # на сколько заглядываем вперед
    draw_window = Config.draw_window  # окно данных
    steps_skip = Config.steps_skip  # шаг сдвига окна данных
    df = functions.get_vwma(df,
                            period_vwma_vfast=period_vwma_vfast,
                            period_vwma_fast=period_vwma_fast,
                            period_vwma_slow=period_vwma_slow)
    data_length = len(df)
    print(f"Data length={data_length} first={df.first_valid_index()} last={df.last_valid_index()}")
    _input = []
    _output = []
    for i in range(df.first_valid_index() + draw_window, df.last_valid_index() - look_forward, steps_skip):
        _result = functions.prepare_data_and_eval(df,
                                                  idx=i,
                                                  window_size=draw_window,
                                                  look_forward=look_forward,
                                                  evaluator=functions.evaluate,
                                                  has_vfast_vwma=True,
                                                  has_fast_vwma=True,
                                                  has_slow_vwma=True)
        if _result is not None:
            _data, _eval = _result
            _input.append(_data)
            _output.append(_eval)

    print(f"Input data contains {len(_input)} elements")
    _input = numpy.array(_input)
    _output = numpy.array(_output)
    end_0 = time.time()
    total_time = end_0 - start
    print(f"Data prepare execution completed in {total_time:.2f} sec")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        _input,
        _output,
        test_size=0.2,
        random_state=42
    )

    end_1 = time.time()
    total_time = end_1 - end_0
    print(f"Data split to train and test data completed in {total_time:.2f} sec")

    # Build the neural network model
    model = Sequential()
    #model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    end_2 = time.time()
    total_time = end_2 - end_1
    print(f"Neural network model compilation  completed in {total_time:.2f} sec")

    callbacks = [
        ModelCheckpoint(functions.join_paths([cur_run_folder, root_folder, "_models", 'cnn_Open{epoch:1d}.hdf5'])),
        # keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        ]
    # Train the model
    _epochs = 99
    history = model.fit(X_train, y_train, epochs=_epochs, batch_size=32, validation_split=0.1, callbacks=callbacks)

    end_3 = time.time()
    total_time = end_3 - end_2
    print(f"Neural network model training  completed in {total_time:.2f} sec")

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    end_4 = time.time()
    total_time = end_4 - end_3
    print(f"Neural network model evaluation  completed in {total_time:.2f} sec")

    end_5 = time.time()
    total_time = end_5 - start
    print(f"Total execution completed in {total_time:.2f} sec")

    functions.show_train_history(history, _epochs, "_test_NN.png")


async def check_nn_with_latest_data():
    root_folder = Config.root_folder

    period_vwma_slow = Config.period_vwma_slow  # период медленной SMA
    period_vwma_fast = Config.period_vwma_fast  # период быстрой SMA
    period_vwma_vfast = Config.period_vwma_vfast  # период супер быстрой SMA
    look_forward = Config.look_forward  # на сколько заглядываем вперед
    draw_window = Config.draw_window  # окно данных

    model = load_model(os.path.join(root_folder, "_models", "cnn_Open99.hdf5"))
    # check architecture
    model.summary()
    connector = aiohttp.TCPConnector(force_close=True, limit=50, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        _ticker = "SiU3"
        _tf = "M1"
        _filename = os.path.join(os.path.join("NN_futures", "csv"), f"{_ticker}_{_tf}.csv")
        old_data = functions.load_candles(_filename)
        new_data = await functions.get_futures_candles(session, _ticker, _tf, start="2023-07-24")
        concat_data = pd.concat([old_data, new_data])
        #print(concat_data)
        full_data = concat_data.drop_duplicates(subset=["datetime"], keep='last').reset_index(drop=True)
        print(full_data)
        data_vwma = functions.get_vwma(full_data, period_vwma_vfast, period_vwma_fast, period_vwma_slow)
        print(data_vwma)
        _last_idx = data_vwma.last_valid_index()
        data_vwma["prediction"] = ""
        data_vwma["real"] = ""
        for idx in range(_last_idx - 100 - look_forward, _last_idx - look_forward):
            _res = functions.prepare_data_and_eval(data_vwma,
                                                   idx=idx,
                                                   window_size=draw_window,
                                                   look_forward=look_forward,
                                                   evaluator=functions.evaluate,
                                                   has_vfast_vwma=True,
                                                   has_fast_vwma=True,
                                                   has_slow_vwma=True)
            if _res is not None:
                _data, _eval = _res
                _predict = model.predict(_data, verbose=0)
                data_vwma.iloc[idx]["prediction"] = _predict
                data_vwma.iloc[idx]["real"] = _eval
            else:
                print(f"Some issue with idx={idx}")

        _result = data_vwma.iloc[_last_idx - 100 - look_forward:_last_idx - look_forward].copy()
        print(f"Prediction and actual:")
        print(_result)
        end = time.time()
        total_time = end - start
        print(f"Total execution completed in {total_time:.2f} sec")
        await connector.close()


#check_nn()
#check_nn2()
#check_nn3()
#check_nn4()
#check_nn5()

#check_data_prep()
#check_my_nn()

loop = asyncio.get_event_loop()  # создаем цикл
task = loop.create_task(  # в цикл добавляем 1 задачу
    check_nn_with_latest_data()
)
loop.run_until_complete(task)



