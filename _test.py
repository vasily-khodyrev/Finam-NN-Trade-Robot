import datetime
import os
import io

import PIL
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance
from PIL import Image
from finam_trade_api.candles import IntraDayCandlesRequestModel, IntraDayInterval
from google.protobuf.json_format import MessageToDict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io

import functions
from FinamPy import FinamPy
from FinamPy.proto.tradeapi.v1.stops_pb2 import STOP_STATUS_ACTIVE

from FinamPy.proto.tradeapi.v1.candles_pb2 import DayCandleTimeFrame, DayCandleInterval, IntradayCandleTimeFrame, \
    IntradayCandleInterval
from google.type.date_pb2 import Date
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from finam_trade_api.client import Client
from finam_trade_api.portfolio.model import PortfolioRequestModel
from finam_trade_api.order.model import (
    BoardType,
    CreateOrderRequestModel,
    CreateStopOrderRequestModel,
    DelOrderModel,
    OrdersRequestModel,
    OrderType,
    PropertyType,
    StopLossModel,
    StopQuantity,
    StopQuantityUnits,
    TakeProfitModel
)


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
    ticker = "SiU3"
    tf = "M10"
    _filename = os.path.join(os.path.join("NN_futures", "csv"), f"{ticker}_{tf}.csv")
    df = pd.read_csv(_filename, sep=',')  # , index_col='datetime')
    if tf in ["M1", "M10", "H1"]:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    dataset = df.tail(5)
    period = 3
    print("Initial data:")
    print(dataset)
    data_with_vwma = functions.get_vwma(dataset, period_vwma_fast=period)
    print(f"Result with period={period}:")
    print(data_with_vwma)


def check_plot_creation():
    ticker = "SiU3"
    tf = "M10"
    _filename = os.path.join(os.path.join("NN_futures", "csv"), f"{ticker}_{tf}.csv")
    df = pd.read_csv(_filename, sep=',')  # , index_col='datetime')
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
    fig2.update_yaxes(range=[0, max_volume * 10], secondary_y=True)
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
    # fig2.show()
    fig2.write_image(file='test.png', format='png', width="512", height="512")
    blank_image = Image.new("RGB", (1024, 1024))
    im = Image.open('test.png')
    blank_image.paste(im, (0, 0))
    blank_image.paste(im, (512, 0))
    blank_image.paste(im, (0, 512))
    blank_image.paste(im, (512, 512))
    blank_image.save("test4.png")


def create_image(df: pd.DataFrame, show_scales: bool) -> PIL.Image:
    hist = df.copy()
    last_date = hist["datetime"].iloc[-1]
    _date = last_date.strftime('%Y-%m-%d %H:%M')
    hist.index = pd.to_datetime(hist['datetime'])
    _data_len = len(hist)
    _width = _height = max(_data_len*4, 1024)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    max_volume = hist['volume'].max()
    fig.add_trace(go.Candlestick(x=hist.index,
                                  open=hist['open'],
                                  high=hist['high'],
                                  low=hist['low'],
                                  close=hist['close'],
                                  ))
    if 'vwma_vfast' in hist.columns:
        if 'vwma_vfast_color' in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index,
                                     y=hist['vwma_vfast'],
                                     mode='markers+lines',
                                     marker={'color': hist['vwma_vfast_color']},
                                     line={'color': 'purple'},
                                     name='VWMA very fast'))
        else:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['vwma_vfast'], marker_color='purple', name='VWMA very fast'))
    if 'vwma_fast' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['vwma_fast'], marker_color='blue', name='VWMA fast'))
    if 'vwma_slow' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['vwma_slow'], marker_color='black', name='VWMA slow'))
    if 'ema_fast' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['ema_fast'], line=dict(color='blue', width=1,
                                                                              dash='dash'), name='EMA'))
    if 'ema_slow' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['ema_slow'], line=dict(color='black', width=1,
                                                                              dash='dash'), name='EMA'))
    fig.add_trace(go.Bar(x=hist.index, y=hist['volume']), secondary_y=True)
    fig.update_yaxes(range=[0, max_volume * 10], secondary_y=True)
    fig.update_xaxes(type="category", categoryorder='category ascending')
    fig.update_yaxes(visible=False, secondary_y=True)
    fig.update_yaxes(gridcolor="black", showgrid=True, side="right")
    fig.update_layout(xaxis_rangeslider_visible=False,
                       autosize=False,
                       width=_width,
                       height=_height,
                       plot_bgcolor='white',
                       showlegend=False)
    if not show_scales:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False},
                           xaxis={'visible': False, 'showticklabels': False})
    else:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))
        fig.update_layout(title_text=f"{_date}")

    fig.show()
    image_data = plotly.io.to_image(fig, width=_width, height=_height, format="png")
    return Image.open(io.BytesIO(image_data))


def check_plot_creation2():
    ticker = "Si"
    tf = "M1"
    _filename = os.path.join(os.path.join("_scan", "csv"), f"futures-{ticker}_{tf}.csv")
    data = functions.load_candles(_filename)
    data_vwma = functions.get_vwma(data, period_vwma_vvfast=10, period_vwma_vfast=50, period_vwma_fast=100, period_vwma_slow=200,
                                   drop_nan=False)
    chart_data = data_vwma.tail(256)
    img = create_image(chart_data, show_scales=True)
    img.save("test_plot2.png")
    pass


def get_levels(df: pd.DataFrame) -> list[float]:
    _first_idx = df.first_valid_index
    _last_idx = df.last_valid_index
    _min = df[["high", "low"]].min()
    _max = df[["high", "low"]].max()
    _n = 30
    _delta = (_max - _min) / _n
    result = [0.0] * _n
    for i in range(_first_idx, _last_idx + 1):
        cur_high = df.at[i, "high"]
        cur_low = df.at[i, "low"]
        cur_volume = df.at[i, "volume"]
        _cur_num = (cur_high - cur_low) / _delta
        volume_part = cur_volume / _cur_num
        cur_idx = (cur_low - _min) // _delta
        j = cur_idx

    pass


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
    # print(df)
    new_df = aggregate(df, 10)
    new_df.to_csv(os.path.join("NN_futures", "csv", "SiU3_M10_2.csv"), index=False, encoding='utf-8', sep=',')
    # print(new_df)
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


def get_account_summary(fp_provider: FinamPy, client_id: str):
    print(f'Учетная запись: {client_id}')
    portfolio = fp_provider.get_portfolio(client_id)  # Получаем портфель
    for position in portfolio.positions:  # Пробегаемся по всем позициям
        print(
            f'  - Позиция ({position.security_code}) {position.balance} @ {position.average_price:.2f} / {position.current_price:.2f}')
    print('  - Позиции + Свободные средства:')
    for currency in portfolio.currencies:
        print(f'    - {currency.balance:.2f} {currency.name}')
    print('  - Свободные средства:')
    for m in portfolio.money:
        print(f'    - {m.balance:.2f} {m.currency}')
    orders = fp_provider.get_orders(client_id).orders  # Получаем заявки
    for order in orders:  # Пробегаемся по всем заявкам
        print(
            f'  - Заявка номер {order.order_no} {"Покупка" if order.buy_sell == "Buy" else "Продажа"} {order.security_board}.{order.security_code} {order.quantity} @ {order.price}')
    stop_orders = fp_provider.get_stops(client_id).stops  # Получаем стоп заявки
    for stop_order in stop_orders:  # Пробегаемся по всем стоп заявкам
        # print(stop_order)
        if stop_order.status == STOP_STATUS_ACTIVE:
            isStopLoss = len(stop_order.stop_loss.ListFields()) > 0
            isTakeProfit = len(stop_order.take_profit.ListFields()) > 0
            stopLossText = f"SL: {stop_order.take_profit.quantity.value} @ {stop_order.stop_loss.activation_price}" if isStopLoss else ""
            takeProfitText = f'TP: {stop_order.take_profit.quantity.value} @ {stop_order.take_profit.activation_price}' if isTakeProfit else ""
            print(
                f'  - Стоп заявка номер {stop_order.stop_id} {"Покупка" if stop_order.buy_sell == "Buy" else "Продажа"} {stop_order.security_board}.{stop_order.security_code} {stopLossText} {takeProfitText}')


def check_account_and_securities():
    FINAM_ACCOUNT = os.environ.get('FINAM_ACCOUNT')
    FINAM_TOKEN = os.environ.get('FINAM_TOKEN')
    fp_provider = FinamPy(FINAM_TOKEN)
    print(f"Checking account {FINAM_ACCOUNT}")
    try:
        get_account_summary(fp_provider, FINAM_ACCOUNT)
        securities = fp_provider.get_securities()
        if securities is not None:
            for security in securities.securities:
                if security.board == "FUT" and security.code == "SiU3":
                    print(
                        f"{security.code} -{security.instrument_code} board {security.board} market {security.market}")
                    print(
                        f'\nИнформация о тикере {security.board}.{security.code} ({security.short_name}, {fp_provider.markets[security.market]}):')
                    print(f'Валюта: {security.currency}')
                    print(f'Лот: {security.lot_size}')
                    decimals = security.decimals  # Кол-во десятичных знаков
                    print(f'Кол-во десятичных знаков: {decimals}')
                    min_step = 10 ** -decimals * security.min_step  # Шаг цены
                    print(f'Шаг цены: {min_step}')
        else:
            print(f"No securities received")
    finally:
        fp_provider.close_channel()


def get_finam_candles():
    FINAM_ACCOUNT = os.environ.get('FINAM_ACCOUNT')
    FINAM_TOKEN = os.environ.get('FINAM_TOKEN')
    fp_provider = FinamPy(FINAM_TOKEN)
    print(f"Checking account {FINAM_ACCOUNT}")
    try:
        time_frame = IntradayCandleTimeFrame.INTRADAYCANDLE_TIMEFRAME_M1
        next_utc_bar_date = fp_provider.msk_to_utc_datetime(datetime.datetime.now() + datetime.timedelta(minutes=2),
                                                            True)
        date_to = Timestamp(seconds=int(next_utc_bar_date.timestamp()), nanos=next_utc_bar_date.microsecond * 1_000)
        interval = IntradayCandleInterval(count=20, to=date_to)
        result = fp_provider.get_intraday_candles("FUT", "SiU3", time_frame, interval)
        if result is not None:
            new_bars_dict = MessageToDict(
                result,
                including_default_value_fields=True)['candles']
            _dates = []
            _opens = []
            _highs = []
            _lows = []
            _closes = []
            _volumes = []
            for new_bar in new_bars_dict:
                _dates.append(
                    fp_provider.utc_to_msk_datetime(datetime.datetime.fromisoformat(new_bar['timestamp'][:-1])))
                _opens.append(round(int(new_bar['open']['num']) * 10 ** -int(new_bar['open']['scale']),
                                    int(new_bar['open']['scale'])))
                _highs.append(round(int(new_bar['high']['num']) * 10 ** -int(new_bar['high']['scale']),
                                    int(new_bar['high']['scale'])))
                _lows.append(round(int(new_bar['low']['num']) * 10 ** -int(new_bar['low']['scale']),
                                   int(new_bar['low']['scale'])))
                _closes.append(round(int(new_bar['close']['num']) * 10 ** -int(new_bar['close']['scale']),
                                     int(new_bar['close']['scale'])))
                _volumes.append(new_bar['volume'])
            _dict = {'datetime': _dates,
                     'open': _opens,
                     'high': _highs,
                     'low': _lows,
                     'close': _closes,
                     'volume': _volumes,
                     }
            res = pd.DataFrame(_dict)
            print(res)
        else:
            print(f"No candles received")
    finally:
        fp_provider.close_channel()


def get_final_candles_v2():
    FINAM_ACCOUNT = os.environ.get('FINAM_ACCOUNT')
    FINAM_TOKEN = os.environ.get('FINAM_VIEW_TOKEN')
    client = Client(FINAM_TOKEN)
    request = IntraDayCandlesRequestModel(
        count=500,
        securityBoard="FUT",
        securityCode="SiU3",
        timeFrame=IntraDayInterval.M1,
        intervalFrom=None,
        intervalTo=None,
    )
    import asyncio

    result = asyncio.run(client.candles.get_in_day_candles(request))
    print(result)


# get_final_candles_v2()
# get_finam_candles()
# check_account_and_securities()
# check_wma()
# check_aggregation()
# check_plot_creation()
check_plot_creation2()
# check_comparators()
