import datetime
import io
import math
import os
import sys

import aiohttp
import aiomoex
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from PIL import Image
from plotly.subplots import make_subplots


def get_timeframe_moex(tf: str):
    """Функция получения типа таймфрейма в зависимости от направления"""
    # - целое число 1 (1 минута), 10 (10 минут), 60 (1 час), 24 (1 день), 7 (1 неделя), 31 (1 месяц) или 4 (1 квартал)
    tfs = {"M1": 1, "M10": 10, "H1": 60, "D1": 24, "W1": 7, "MN1": 31, "Q1": 4}
    if tf in tfs: return tfs[tf]
    return False


def transform_moex_to_needed_tf(tf: str, aggregation: int, src_df: pd.DataFrame) -> pd.DataFrame:
    return 0


def get_future_key(key, tf, future_tf):
    """Высчитываем следующий key для старшего ТФ, кроме tf == D1, W1, MN1 и кроме future_tf == W1, MN1"""
    if tf in ["D1", "W1", "MN1"] or future_tf in ["W1", "MN1"]: return False

    _hour = key.hour
    _minute = key.minute
    if future_tf not in ["D1", "W1", "MN1"]:
        future_key = datetime.datetime.fromisoformat(key.strftime('%Y-%m-%d') + f" {key.hour:02d}:00")
    else:
        future_key = datetime.datetime.fromisoformat(key.strftime('%Y-%m-%d') + " 00:00")

    tfs = {'M1': 1, 'M2': 2, 'M5': 5, 'M10': 10, 'M15': 15, 'M30': 30, 'H1': 60, 'H2': 120, 'H4': 240, 'D1': 1440,
           'W1': False, 'MN1': False}

    _k = tfs[tf]
    _k2 = tfs[future_tf]
    _i1 = _minute // _k2

    future_key = future_key + datetime.timedelta(minutes=_k2 * (_i1 + 1))
    future_key2 = future_key + datetime.timedelta(minutes=_k2 * (_i1 + 1))
    # print(key, "=>", future_key, "=>", future_key2, f"\t{tf} => {future_tf}")

    # print("\t", _hour, _minute, _k, _k2, _i1)
    return key, future_key, future_key2


def detect_class(key, future_key, future_key2, arr_OHLCV_1, timeframe_1, expected_change):
    """определяем класс к которому относятся future свечи на future_key, future_key2  """
    if future_key in arr_OHLCV_1:
        _future_ohlcv = arr_OHLCV_1[future_key]
        # print(_future_ohlcv, "111111", key, "=>", future_key)
    else:
        # ищем ближайший future_key
        for k in list(arr_OHLCV_1.keys()):
            if k > key:
                future_key = k
                break
        _future_ohlcv = arr_OHLCV_1[future_key]
        # print(_future_ohlcv, "22222", key, "=>", future_key)

    # print(_future_ohlcv, "*******")
    _percent_OC = _future_ohlcv[5]  # 5 == _percent_OC
    _sign = math.copysign(1, _percent_OC)  # берем знак процента
    # print(_percent_OC, _sign)
    _classification_percent = _sign * get_classification(abs(_percent_OC), tf=timeframe_1, ex_ch=expected_change)

    if _classification_percent == 0:
        # попытка ещё раз сделать классификацию, заглянув на 2 свечи вперед
        _pre_percent_OC = _percent_OC  # учитываем % и предыдущей свечи
        future_key = future_key2
        if future_key in arr_OHLCV_1:
            _future_ohlcv = arr_OHLCV_1[future_key]
            # print(_future_ohlcv, "33333", key, "=>", future_key)
        else:
            # ищем ближайший future_key
            for k in list(arr_OHLCV_1.keys()):
                if k > key:
                    future_key = k
                    break
            _future_ohlcv = arr_OHLCV_1[future_key]
            # print(_future_ohlcv, "44444", key, "=>", future_key)
        # print(_future_ohlcv, "**222**")
        _percent_OC = _future_ohlcv[5]  # 5 == _percent_OC
        _sign = math.copysign(1, _percent_OC)  # берем знак процента
        # print(_percent_OC, _sign)
        _percent_OC += _pre_percent_OC  # учитываем % и предыдущей свечи
        _classification_percent = _sign * get_classification(abs(_percent_OC), tf=timeframe_1, ex_ch=expected_change)

    return _classification_percent


def get_classification(_p, tf, ex_ch):
    """Определяем класс по проценту свечи"""
    _class_percent = 6
    for i in range(len(ex_ch[tf]) - 1):
        if ex_ch[tf][i] <= _p < ex_ch[tf][i + 1]:
            _class_percent = i
            break
    return _class_percent


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error_and_exit(_error, _error_code):
    '''Функция вывода ошибки и остановки программы'''
    print(bcolors.FAIL + _error + bcolors.ENDC)
    exit(_error_code)


def print_warning(_warning):
    '''Функция вывода предупреждения'''
    print(bcolors.WARNING + _warning + bcolors.ENDC)


def join_paths(paths):
    """Функция формирует путь из списка"""
    _folder = ''
    for _path in paths:
        _folder = os.path.join(_folder, _path)
    return _folder


def create_some_folders(timeframes, root_folder='NN', classes=None):
    """Функция создания необходимых директорий"""
    folder = os.path.join(root_folder, f"NN_winner")
    if not os.path.exists(folder): os.makedirs(folder)

    folder = os.path.join(root_folder, f"csv")
    if not os.path.exists(folder): os.makedirs(folder)

    folder = root_folder
    if not os.path.exists(folder): os.makedirs(folder)

    for timeFrame in timeframes:
        _folder = os.path.join(folder, f"training_dataset_{timeFrame}")
        if not os.path.exists(_folder): os.makedirs(_folder)

        if classes:
            for _class in classes:
                _folder_class = os.path.join(_folder, f"{_class}")
                if not os.path.exists(_folder_class): os.makedirs(_folder_class)

    _folder = os.path.join(folder, f"_data")
    if not os.path.exists(_folder): os.makedirs(_folder)

    _folder = os.path.join(folder, f"_models")
    if not os.path.exists(_folder): os.makedirs(_folder)


def start_redirect_output_from_screen_to_file(redirect, filename):
    '''Функция старта перенаправления вывода с консоли в файл'''
    if redirect:
        sys.stdout = open(filename, 'w', encoding='utf8')


def stop_redirect_output_from_screen_to_file():
    '''Функция прекращения перенаправления вывода с консоли в файл'''
    try:
        sys.stdout.close()
    except:
        pass


async def get_futures_candles(session, ticker: str, timeframe: str, start: str, end: str):
    """Get candles for FUTURES from MOEX."""
    tf = get_timeframe_moex(timeframe)
    data = await aiomoex.get_market_candles(
        session,
        ticker,
        interval=tf,
        start=start,
        end=end,
        market="forts",
        engine="futures"
    )
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
    # для M1, M10, H1 - приводим дату свечи в правильный вид
    if tf in [1, 10, 60]:
        df['datetime'] = df['datetime'].apply(lambda x: x + datetime.timedelta(minutes=tf))
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    return df


async def get_stock_candles(session: aiohttp.ClientSession, ticker: str, timeframe: str, start: str, end: str):
    """Get stock candles for STOCKS from MOEX."""
    tf = get_timeframe_moex(timeframe)
    data = await aiomoex.get_market_candles(session, ticker, interval=tf, start=start, end=end)  # M10
    df = pd.DataFrame(data)
    #Check if data is present - return empty dataframe with columns
    if df.empty:
        return pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
    # для M1, M10, H1 - приводим дату свечи в правильный вид
    if tf in [1, 10, 60]:
        df['datetime'] = df['datetime'].apply(lambda x: x + datetime.timedelta(minutes=tf))
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    return df


async def get_stock_candles_s(ticker: str, timeframe: str, start: str, end: str):
    """Get stock candles for STOCKS from MOEX. New session is created"""
    with aiohttp.ClientSession() as session:
        return get_stock_candles(session, ticker, timeframe, start, end)


def aggregate(df: pd.DataFrame, period_min: int) -> pd.DataFrame:
    """Aggregate Функция получения стоковых свечей с MOEX."""
    result = df.copy()
    result['index_datetime'] = result['datetime'].apply(lambda x: x - datetime.timedelta(minutes=1))
    result['datetime'] = pd.to_datetime(result['index_datetime'])
    result["close"] = result["close"].astype(float)
    result["open"] = result["open"].astype(float)
    result["high"] = result["high"].astype(float)
    result["volume"] = result["volume"].astype(float)
    result.set_index('index_datetime', inplace=True)
    freq = '' + str(period_min) + 'min'
    result = result.groupby(pd.Grouper(freq=freq)).agg({'datetime': lambda x: __get_period_lambda(x, period_min),
                                                        'open': 'first',
                                                        'high': 'max',
                                                        'low': 'min',
                                                        'close': 'last',
                                                        'volume': 'sum'})
    result['datetime'] = result['datetime'].apply(lambda x: x + datetime.timedelta(minutes=period_min))
    result.dropna(inplace=True)  # удаляем все NULL значения
    result = result[["datetime", "open", "high", "low", "close", "volume"]].copy()
    return result


def __get_period_lambda(x, period):
    if len(x) == 0:
        return np.nan

    tm = x[0]
    if period <= 60:
        tm = tm - datetime.timedelta(minutes=tm.minute % period,
                                     seconds=tm.second,
                                     microseconds=tm.microsecond)
    else:
        hours_div = period // 60
        tm = tm - datetime.timedelta(hours=tm.hour % hours_div,
                                     minutes=tm.minute,
                                     seconds=tm.second,
                                     microseconds=tm.microsecond)
    return tm


def get_vwma(df_in: pd.DataFrame, period_vwma_vfast: int, period_vwma_fast: int, period_vwma_slow: int) -> pd.DataFrame:
    """Calculate Volume Weighted Moving Average"""
    df = df_in.copy()
    df['cv'] = df['close'] * df['volume']

    df['cv_sum_vfast'] = df['cv'].rolling(period_vwma_vfast).sum()
    df['volume_sum_vfast'] = df['volume'].rolling(period_vwma_vfast).sum()
    df['vwma_vfast'] = df['cv_sum_vfast'] / df['volume_sum_vfast']  # формируем VWMA very fast

    df['cv_sum_fast'] = df['cv'].rolling(period_vwma_fast).sum()
    df['volume_sum_fast'] = df['volume'].rolling(period_vwma_fast).sum()
    df['vwma_fast'] = df['cv_sum_fast'] / df['volume_sum_fast'] # формируем VWMA fast

    df['cv_sum_slow'] = df['cv'].rolling(period_vwma_slow).sum()
    df['volume_sum_slow'] = df['volume'].rolling(period_vwma_slow).sum()
    df['vwma_slow'] = df['cv_sum_slow'] / df['volume_sum_slow'] # формируем VWMA slow
    df.dropna(inplace=True)  # удаляем все NULL значения
    return df[["datetime", "open", "high", "low", "close", "volume", "vwma_vfast", "vwma_fast", "vwma_slow"]].copy()


def create_image(df: pd.DataFrame, show_scales: bool) -> Image:
    hist = df.copy()
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    max_volume = hist['volume'].max()
    fig2.add_trace(go.Candlestick(x=hist.index,
                                  open=hist['open'],
                                  high=hist['high'],
                                  low=hist['low'],
                                  close=hist['close'],
                                  ))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_vfast'], marker_color='purple', name='VWMA very fast'))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_fast'], marker_color='blue', name='VWMA fast'))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_slow'], marker_color='black', name='VWMA slow'))
    fig2.add_trace(go.Bar(x=hist.index, y=hist['volume']), secondary_y=True)
    fig2.update_yaxes(range=[0, max_volume*10], secondary_y=True)
    fig2.update_yaxes(visible=False, secondary_y=True)
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                       xaxis_rangeslider_visible=False,
                       autosize=False,
                       width=512,
                       height=512,
                       plot_bgcolor='white',
                       showlegend=False)
    if not show_scales:
        fig2.update_layout(yaxis={'visible': False, 'showticklabels': False},
                           xaxis={'visible': False, 'showticklabels': False})
    #fig2.show()
    image_data = plotly.io.to_image(fig2, width=512, height=512, format="png")
    return Image.open(io.BytesIO(image_data))
