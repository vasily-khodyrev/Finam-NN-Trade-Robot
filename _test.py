import datetime
import random

from PIL import Image
import numpy as np
import pandas as pd

import yfinance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import functions
import functions_nn
import os
import matplotlib.pyplot as plt
from my_config.trade_config import Config  # Файл конфигурации торгового робота


def wma(df, period_sma_fast, period_sma_slow):
    df['cv'] = df['Close'] * df['Volume']
    df['cv_sum_fast'] = df['cv'].rolling(period_sma_fast).sum()
    df['volume_sum_fast'] = df['Volume'].rolling(period_sma_fast).sum()
    df['vwma_fast'] = df['cv_sum_fast'] / df['volume_sum_fast']
    df['cv_sum_slow'] = df['cv'].rolling(period_sma_slow).sum()
    df['volume_sum_slow'] = df['Volume'].rolling(period_sma_slow).sum()
    df['vwma_slow'] = df['cv_sum_slow'] / df['volume_sum_slow']
    return df[["Open", "High", "Low", "Close", "Volume", "vwma_fast", "vwma_slow"]].copy()


def check_wma():
    df = pd.DataFrame({'close':np.arange(11), 'volume': random.randrange(1, 11)})
    df = wma(df, 1, 2)
    print(df)


def check_plot_creation():
    tsla = yfinance.Ticker('TSLA')
    hist = tsla.history(period='1y')
    hist = wma(hist, 10, 20)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    max_volume = hist['Volume'].max()
    fig2.add_trace(go.Candlestick(x=hist.index,
                                  open=hist['Open'],
                                  high=hist['High'],
                                  low=hist['Low'],
                                  close=hist['Close'],
                                  ))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_fast'], marker_color='blue', name='10 Day MA'))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_slow'], marker_color='black', name='20 Day MA'))
    fig2.add_trace(go.Bar(x=hist.index, y=hist['Volume']), secondary_y=True)
    fig2.update_yaxes(range=[0, max_volume*10], secondary_y=True)
    fig2.update_yaxes(visible=False, secondary_y=True)
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                       yaxis={'visible': False, 'showticklabels': False},
                       xaxis={'visible': False, 'showticklabels': False},
                       xaxis_rangeslider_visible=False,
                       autosize=False,
                       width=512,
                       height=512,
                       plot_bgcolor='white',
                       showlegend=False)
    #fig2.show()
    fig2.write_image(file='test.png', format='png', width="512", height="512")
    blank_image = Image.new("RGB", (1024, 1024))
    im = Image.open('test.png')
    blank_image.paste(im, (0, 0))
    blank_image.paste(im, (512, 0))
    blank_image.paste(im, (0, 512))
    blank_image.paste(im, (512, 512))
    blank_image.save("test4.png")


def get_period_lambda(x, period):
    if len(x) == 0:
        return np.nan

    tm = x[0]
    tm = tm - datetime.timedelta(minutes=tm.minute % period,
                                 seconds=tm.second,
                                 microseconds=tm.microsecond)
    return tm


def aggregate(df, period) -> pd.DataFrame:
    result = df.copy()
    result['index_datetime'] = result['datetime'].apply(lambda x: x - datetime.timedelta(minutes=1))
    result['datetime'] = pd.to_datetime(result['index_datetime'])
    result["close"] = result["close"].astype(float)
    result["open"] = result["open"].astype(float)
    result["high"] = result["high"].astype(float)
    result["volume"] = result["volume"].astype(float)
    result.set_index('index_datetime', inplace=True)
    freq = '' + str(period) + 'min'
    result = result.groupby(pd.Grouper(freq=freq)).agg({'datetime': lambda x: get_period_lambda(x, period),
                                                        'open': 'first',
                                                        'high': 'max',
                                                        'low': 'min',
                                                        'close': 'last',
                                                        'volume': 'sum'})
    print(result.columns)
    result['datetime'] = result['datetime'].apply(lambda x: x + datetime.timedelta(minutes=period))
    result.dropna(inplace=True)  # удаляем все NULL значения
    return result


def check_aggregation():
    df = pd.DataFrame([[datetime.datetime(year=2023, month=7, day=25, hour=12, minute=1), 1, 0, 2, 0, 1],
                       [datetime.datetime(year=2023, month=7, day=25, hour=12, minute=2), 2, 1, 3, 0, 1],
                       [datetime.datetime(year=2023, month=7, day=25, hour=12, minute=3), 3, 2, 4, 1, 1],
                       [datetime.datetime(year=2023, month=7, day=25, hour=12, minute=4), 4, 3, 5, 2, 1],
                       [datetime.datetime(year=2023, month=7, day=25, hour=12, minute=5), 7, 4, 9, 3, 1],
                       [datetime.datetime(year=2023, month=7, day=26, hour=12, minute=1), 4, 3, 5, 2, 1],
                       [datetime.datetime(year=2023, month=7, day=26, hour=12, minute=2), 7, 4, 9, 3, 1],
                       [datetime.datetime(year=2023, month=7, day=26, hour=12, minute=3), 4, 3, 5, 2, 1],
                       [datetime.datetime(year=2023, month=7, day=26, hour=12, minute=4), 7, 4, 9, 3, 1],
                       [datetime.datetime(year=2023, month=7, day=26, hour=12, minute=5), 7, 4, 9, 3, 1],
                       ],
                      columns=['datetime', 'close', 'open', 'high', 'low', 'volume'])
    _filename = os.path.join(os.path.join("NN_futures", "csv"), "SiU3_M1.csv")
    df = pd.read_csv(_filename, sep=',')  # , index_col='datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    #print(df)
    new_df = aggregate(df, 10)
    new_df.to_csv(os.path.join("NN_futures", "csv", "SiU3_M10_2.csv"), index=False, encoding='utf-8', sep=',')
    #print(new_df)
    pass


class TestData:
    def __init__(self, a: str, b: str, c: str):
        self._a = a
        self._b = b
        self._c = c

    def get_a(self) -> str:
        return self._a

    def get_b(self) -> str:
        return self._b

    def get_c(self) -> str:
        return self._c

    def __repr__(self):
        return f"({self._a},{self._b},{self._c})"


def check_comparators():
    array = [TestData('a', 'B', '6'), TestData('a', 'A', '7'), TestData('b', 'a', '8'), TestData('a', 'A', '9')]
    print(array)
    res = sorted(array, key=lambda a: (a.get_a(), a.get_b(), a.get_c()), reverse=True)
    print(res)
    pass


#check_wma()
#check_aggregation()
#check_plot_creation()
check_comparators()


