import datetime
import io
import math
import os
import sys
import time
from typing import Optional, Tuple, Callable

import aiohttp
import aiomoex
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from FinamPy import FinamPy
from FinamPy.proto.tradeapi.v1.candles_pb2 import IntradayCandleTimeFrame, IntradayCandleInterval
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict


def get_timeframe_moex(tf: str):
    """Функция получения типа таймфрейма в зависимости от направления"""
    # - целое число 1 (1 минута), 10 (10 минут), 60 (1 час), 24 (1 день), 7 (1 неделя), 31 (1 месяц) или 4 (1 квартал)
    tfs = {"M1": 1, "M10": 10, "H1": 60, "D1": 24, "W1": 7, "MN1": 31, "Q1": 4}
    if tf in tfs: return tfs[tf]
    return False


def get_timeframe_finam(tf: str) -> Optional[IntradayCandleTimeFrame]:
    """Функция получения типа таймфрейма в зависимости от направления (для FINAM)"""
    tfs = {"M1": IntradayCandleTimeFrame.INTRADAYCANDLE_TIMEFRAME_M1,
           "M5": IntradayCandleTimeFrame.INTRADAYCANDLE_TIMEFRAME_M5,
           "M15": IntradayCandleTimeFrame.INTRADAYCANDLE_TIMEFRAME_M15,
           "H1": IntradayCandleTimeFrame.INTRADAYCANDLE_TIMEFRAME_H1}
    if tf in tfs:
        return tfs[tf]
    return None


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


async def get_futures_candles(session,
                              ticker: str,
                              timeframe: str,
                              start: str,
                              end: Optional[str] = None,
                              file_store: Optional[str] = None):
    """Get candles for FUTURES from MOEX."""
    old_data = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    if file_store is not None:
        file_exists = os.path.isfile(file_store)
        if file_exists:  # Если файл существует
            old_data = load_candles(file_store)
            if len(old_data) > 0:
                first_date = old_data["datetime"].iloc[0]
                last_date = old_data["datetime"].iloc[-1]
                requested_start = datetime.datetime.strptime(start, '%Y-%m-%d')
                if first_date >= requested_start:
                    old_data = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
                else:
                    #always request from last date in file to keep it solid
                    start = last_date.strftime("%Y-%m-%d")

    tf = get_timeframe_moex(timeframe)
    _get_moex_data = True
    if not old_data.empty:
        _last_date = old_data["datetime"].iloc[-1]
        _now = datetime.datetime.now()
        _delta_min = int((_now - _last_date).total_seconds() // 60)
        if tf == 1:
            _get_moex_data = _delta_min > 15
        if tf == 60:
            _get_moex_data = _delta_min > 60

    if _get_moex_data:
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
        if df.empty:
            df = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    else:
        df = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    # Check if data is present - return empty dataframe with columns
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
        # для M1, M10, H1 - приводим дату свечи в правильный вид
        if tf in [1, 10, 60]:
            df['datetime'] = df['datetime'].apply(lambda x: x + datetime.timedelta(minutes=tf))
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    full_data = df
    if file_store is not None:
        concat_data = pd.concat([old_data, df])
        full_data = concat_data.drop_duplicates(subset=["datetime"], keep='last').reset_index(drop=True)
    if tf == 1:
        _last_date = full_data["datetime"].iloc[-1]
        _now = datetime.datetime.now()
        _delta_min = int((_now - _last_date).total_seconds() // 60)
        if _delta_min > 0:
            print(f"{ticker}-{timeframe} MOEX delta is {_delta_min} min")
            _latest_data = get_finam_futures_intraday_candles(ticker, timeframe, _delta_min + 5)
            concat_data = pd.concat([full_data, _latest_data])
            full_data = concat_data.drop_duplicates(subset=["datetime"], keep='last').reset_index(drop=True)
    if file_store is not None:
        store_candles(full_data, file_store)
    return full_data


def get_finam_futures_intraday_candles(ticker: str, tf: str, candles_count: int = 500) -> pd.DataFrame:
    start = time.time()
    FINAM_TOKEN = os.environ.get('FINAM_VIEW_TOKEN')
    fp_provider = FinamPy(FINAM_TOKEN)
    result = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    try:
        time_frame = get_timeframe_finam(tf)
        next_utc_bar_date = fp_provider.msk_to_utc_datetime(datetime.datetime.now() + datetime.timedelta(minutes=2), True)
        date_to = Timestamp(seconds=int(next_utc_bar_date.timestamp()), nanos=next_utc_bar_date.microsecond * 1_000)
        interval = IntradayCandleInterval(count=candles_count, to=date_to)
        _result = fp_provider.get_intraday_candles("FUT", ticker, time_frame, interval)
        if _result is not None:
            new_bars_dict = MessageToDict(
                _result,
                including_default_value_fields=True)['candles']
            _dates = []
            _opens = []
            _highs = []
            _lows = []
            _closes = []
            _volumes = []
            for new_bar in new_bars_dict:
                _dates.append(fp_provider.utc_to_msk_datetime(datetime.datetime.fromisoformat(new_bar['timestamp'][:-1])))
                _opens.append(round(int(new_bar['open']['num']) * 10 ** -int(new_bar['open']['scale']),
                                    int(new_bar['open']['scale'])))
                _highs.append(round(int(new_bar['high']['num']) * 10 ** -int(new_bar['high']['scale']),
                                    int(new_bar['high']['scale'])))
                _lows.append(round(int(new_bar['low']['num']) * 10 ** -int(new_bar['low']['scale']),
                                   int(new_bar['low']['scale'])))
                _closes.append(round(int(new_bar['close']['num']) * 10 ** -int(new_bar['close']['scale']),
                                     int(new_bar['close']['scale'])))
                _volumes.append(int(new_bar['volume']))
            _dict = {'datetime': _dates,
                     'open': _opens,
                     'high': _highs,
                     'low': _lows,
                     'close': _closes,
                     'volume': _volumes,
                     }
            result = pd.DataFrame(_dict)
            _moex_tf = get_timeframe_moex(tf)
            if _moex_tf in [1, 10, 60]:
                result['datetime'] = result['datetime'].apply(lambda x: x + datetime.timedelta(minutes=_moex_tf))
        else:
            print(f"No candles received for {ticker}")
    finally:
        fp_provider.close_channel()
    end_0 = time.time()
    total_time = end_0 - start
    print(f"get_finam_futures_intraday_candles {ticker} - {tf} - {candles_count} candles -  completed in {total_time:.2f} sec")
    return result


async def get_stock_candles(session: aiohttp.ClientSession,
                            ticker: str,
                            timeframe: str,
                            start: str,
                            end: Optional[str] = None,
                            file_store: Optional[str] = None):
    """Get stock candles for STOCKS from MOEX."""
    tf = get_timeframe_moex(timeframe)
    old_data = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    if file_store is not None:
        file_exists = os.path.isfile(file_store)
        if file_exists:  # Если файл существует
            old_data = load_candles(file_store)
            if len(old_data) > 0:
                first_date = old_data["datetime"].iloc[0]
                last_date = old_data["datetime"].iloc[-1]
                requested_start = datetime.datetime.strptime(start, '%Y-%m-%d')
                if first_date >= requested_start:
                    #drop stored data since no more can be used
                    old_data = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
                else:
                    #if requested_start < last_date:
                    #always request data from latest in file to keep it solid
                    start = last_date.strftime("%Y-%m-%d")
    data = await aiomoex.get_market_candles(session, ticker, interval=tf, start=start, end=end)  # M10
    df = pd.DataFrame(data)
    # Check if data is present - return empty dataframe with columns
    if df.empty:
        df = pd.DataFrame(data=None, columns=["datetime", "open", "high", "low", "close", "volume"])
    else:
        df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
        # для M1, M10, H1 - приводим дату свечи в правильный вид
        if tf in [1, 10, 60]:
            df['datetime'] = df['datetime'].apply(lambda x: x + datetime.timedelta(minutes=tf))
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    full_data = df
    if file_store is not None:
        concat_data = pd.concat([old_data, df])
        full_data = concat_data.drop_duplicates(subset=["datetime"], keep='last').reset_index(drop=True)
        store_candles(full_data, file_store)
    return full_data


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


def get_vwma(df_in: pd.DataFrame,
             period_vwma_vfast: Optional[int] = None,
             period_vwma_fast: Optional[int] = None,
             period_vwma_slow: Optional[int] = None,
             period_ema_fast: Optional[int] = None,
             period_ema_slow: Optional[int] = None,
             drop_nan: Optional[bool] = True) -> pd.DataFrame:
    """Calculate Volume Weighted Moving Average"""
    df = df_in.copy()
    df['cv'] = df['close'] * df['volume']

    result_columns = ["datetime", "open", "high", "low", "close", "volume"]

    if period_vwma_vfast is not None:
        df['cv_sum_vfast'] = df['cv'].rolling(period_vwma_vfast).sum()
        df['volume_sum_vfast'] = df['volume'].rolling(period_vwma_vfast).sum()
        df['vwma_vfast'] = df['cv_sum_vfast'] / df['volume_sum_vfast']  # формируем VWMA very fast
        result_columns.append("vwma_vfast")

    if period_vwma_fast is not None:
        df['cv_sum_fast'] = df['cv'].rolling(period_vwma_fast).sum()
        df['volume_sum_fast'] = df['volume'].rolling(period_vwma_fast).sum()
        df['vwma_fast'] = df['cv_sum_fast'] / df['volume_sum_fast']  # формируем VWMA fast
        result_columns.append("vwma_fast")

    if period_vwma_slow is not None:
        df['cv_sum_slow'] = df['cv'].rolling(period_vwma_slow).sum()
        df['volume_sum_slow'] = df['volume'].rolling(period_vwma_slow).sum()
        df['vwma_slow'] = df['cv_sum_slow'] / df['volume_sum_slow']  # формируем VWMA slow
        result_columns.append("vwma_slow")
    if period_ema_fast is not None:
        df['ema_fast'] = df['close'].rolling(period_ema_fast).mean()
        result_columns.append("ema_fast")
    if period_ema_slow is not None:
        df['ema_slow'] = df['close'].rolling(period_ema_slow).mean()
        result_columns.append("ema_slow")
    if drop_nan:
        df.dropna(inplace=True)  # удаляем все NULL значения
    return df[result_columns].reset_index(drop=True).copy()


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
    if 'vwma_vfast' in hist.columns:
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_vfast'], marker_color='purple', name='VWMA very fast'))
    if 'vwma_fast' in hist.columns:
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_fast'], marker_color='blue', name='VWMA fast'))
    if 'vwma_slow' in hist.columns:
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['vwma_slow'], marker_color='black', name='VWMA slow'))
    if 'ema_fast' in hist.columns:
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['ema_fast'], line=dict(color='blue', width=1,
                                                                              dash='dash'), name='EMA'))
    if 'ema_slow' in hist.columns:
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['ema_slow'], line=dict(color='black', width=1,
                                                                              dash='dash'), name='EMA'))
    fig2.add_trace(go.Bar(x=hist.index, y=hist['volume']), secondary_y=True)
    fig2.update_yaxes(range=[0, max_volume * 10], secondary_y=True)
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
    # fig2.show()
    image_data = plotly.io.to_image(fig2, width=512, height=512, format="png")
    return Image.open(io.BytesIO(image_data))


def load_candles(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    return df


def store_candles(data: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False, encoding='utf-8', sep=',', date_format='%Y-%m-%d %H:%M:%S')


def evaluate_up(df: pd.DataFrame,
                idx: int,
                len_check: int,
                max_profit: float = 0.003,
                max_loss: float = 0.001,
                ) -> float:
    """
        Evaluates next candle -
        Return: In case of UP - 1.0
                In other cases [0,0]
    """
    _max_idx = df.last_valid_index()
    _max_len = len_check
    _max_profit = max_profit
    _max_loss = max_loss
    prev_close = df.at[idx, "close"]
    up_max_profit_price = prev_close * (1 + _max_profit)
    up_max_loss_price = prev_close * (1 - _max_loss)
    down_max_profit_price = prev_close * (1 - _max_profit)
    down_max_loss_price = prev_close * (1 + _max_loss)
    no_up = False
    for i in range(idx, idx + _max_len):
        if i > _max_idx:
            break
        cur_close = df.at[i, "close"]
        if no_up:
            break
        if not no_up and cur_close <= prev_close:
            no_up = True
    if not no_up:
        return 1.0
    return 0.0

def evaluate_up_down(df: pd.DataFrame,
                     idx: int,
                     len_check: int,
                     max_profit: float = 0.003,
                     max_loss: float = 0.001,
                     ) -> list[float]:
    """
        Evaluates next candle -
          if its close is higher - return [1.0, 0.0]
          if its close is lower - return [0.0, 1.0]
        Return: In case of UP - [1,0]
                In case of DOWN - [0,1]
                In other cases [0,0]
    """
    _max_idx = df.last_valid_index()
    _max_len = len_check
    _max_profit = max_profit
    _max_loss = max_loss
    prev_close = df.at[idx, "close"]
    up_max_profit_price = prev_close * (1 + _max_profit)
    up_max_loss_price = prev_close * (1 - _max_loss)
    down_max_profit_price = prev_close * (1 - _max_profit)
    down_max_loss_price = prev_close * (1 + _max_loss)
    no_up = False
    no_down = False
    for i in range(idx, idx + _max_len):
        if i > _max_idx:
            break
        cur_close = df.at[i, "close"]
        if no_up and no_down:
            break
        if not no_up and cur_close < prev_close:
            no_up = True
        if not no_down and cur_close > prev_close:
            no_down = True
    if not no_up:
        return [1.0, 0.0]
    if not no_down:
        return [0.0, 1.0]
    return [0.0, 0.0]


def evaluate_tp_sl(df: pd.DataFrame,
                   idx: int,
                   len_check: int,
                   max_profit: float = 0.003,
                   max_loss: float = 0.001,
                   ) -> list[float]:
    """
        Evaluates data starting from idx and say whether "close" data will go Up or Down
        (with provided max_profit and loss criteria)
        len_check - how many next items should be checked (at max - since event may happen earlier)
        Return: In case of UP - [1,0]
                In case of DOWN - [0,1]
                In other cases [0,0]
    """
    _max_idx = df.last_valid_index()
    _max_len = len_check
    _max_profit = max_profit
    _max_loss = max_loss
    prev_close = df.at[idx - 1, "close"]
    up_max_profit_price = prev_close * (1 + _max_profit)
    up_max_loss_price = prev_close * (1 - _max_loss)
    down_max_profit_price = prev_close * (1 - _max_profit)
    down_max_loss_price = prev_close * (1 + _max_loss)
    no_up = False
    no_down = False
    for i in range(idx, idx + _max_len):
        if i > _max_idx:
            return [0.0, 0.0]

        cur_high = df.at[i, "high"]
        cur_low = df.at[i, "low"]
        if no_up and no_down:
            return [0.0, 0.0]
        if not no_up and cur_low <= up_max_loss_price:
            no_up = True
        if not no_down and cur_high >= down_max_loss_price:
            no_down = True
        if not no_up and cur_high > up_max_profit_price:
            return [1.0, 0.0]
        if not no_down and cur_low < down_max_profit_price:
            return [0.0, 1.0]
    return [0.0, 0.0]


def prepare_data_and_eval(df: pd.DataFrame,
                          idx: int,
                          window_size: int,
                          look_forward: int,
                          evaluator: Callable[[pd.DataFrame, int, int], list[float]],
                          has_vfast_vwma: Optional[bool] = False,
                          has_fast_vwma: Optional[bool] = False,
                          has_slow_vwma: Optional[bool] = False,
                          flatten_data: Optional[bool] = True,
                          scaler: Optional[MinMaxScaler] = None
                          ) -> Optional[Tuple[list[float], list[float]]]:
    """
        Provides prepared data for neural network:
           - Gets "window_size" rows till "idx" in DataFrame
           - Normalizes values (divide by max open value)
           - Gets evaluation in the future (using "evaluator" function)
           - has_vfast_vwma, has_fast_vwma, has_slow_vwma indicates on whether we need to include
           corresponding vwma values in the result
        Returns:
            Array with 2 items: normalized data + evaluation
            or None in case there's not enough data starting from index "idx"

    """
    values = prepare_data(df, idx, window_size, has_vfast_vwma, has_fast_vwma, has_slow_vwma, scaler, flatten_data)
    if values is None:
        return None
    _eval = evaluator(df, idx + 1, look_forward)
    return values, _eval


def prepare_data(df: pd.DataFrame,
                 idx: int,
                 window_size: int,
                 has_vfast_vwma: Optional[bool] = False,
                 has_fast_vwma: Optional[bool] = False,
                 has_slow_vwma: Optional[bool] = False,
                 scaler: Optional[MinMaxScaler] = None,
                 flatten_data: Optional[bool] = True,
                 ) -> Optional[list[float]]:
    """
        Provides prepared data for neural network:
           - Gets "window_size" rows starting from "idx" in DataFrame
           - Normalizes values (divide by max open value)
           - has_vfast_vwma, has_fast_vwma, has_slow_vwma indicates on whether we need to include
           corresponding vwma values in the result
        Returns:
            Array of normalized data or None in case there's not enough data starting from index "idx"

    """
    # print(f"select range from {idx} to {idx + window_size - 1}")
    # todo seems volume should be removed
    # columns = ["open", "close", "low", "high", "volume"]
    columns = ["open", "close", "low", "high"]
    if has_vfast_vwma:
        columns.append("vwma_vfast")
    if has_fast_vwma:
        columns.append("vwma_fast")
    if has_slow_vwma:
        columns.append("vwma_slow")
    # print(df)
    dfRange = df[columns].iloc[:idx + 1].tail(window_size).copy()
    if len(dfRange) != window_size:
        return None
    if scaler is not None:
        dfRange = scaler.transform(dfRange)
    # print(f"select range from {idx} to {idx + window_size - 1} AFTER NORMALIZATION")
    # print(dfRange)
    values = dfRange
    if flatten_data:
        if isinstance(dfRange, pd.DataFrame):
            values = dfRange.values.flatten().tolist()
        elif isinstance(dfRange, numpy.ndarray):
            values = dfRange.reshape(1, -1).tolist()[0]
        else:
            print("Data not supported!")
            values = None
    return values


# Print iterations progress
def printProgressBar(iteration: int,
                     total: int,
                     prefix: str = '',
                     suffix: str = '',
                     decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def show_train_history(history, epochs: int, filename: Optional[str]):
    # графики потерь и точности на обучающих и проверочных наборах
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if filename is not None:
        plt.savefig(filename, dpi=150)
    plt.show()
