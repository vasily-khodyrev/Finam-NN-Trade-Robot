import asyncio
import datetime
import math
import os
import time
import json
import signal
import sys
from enum import Enum
from typing import Optional, List
import telebot

import aiohttp
import aiomoex
import chevron
import pandas as pd
from PIL import Image

import functions


class ISSSecurity:
    def __init__(self,
                 ticker: str,
                 name: str,
                 list_level: Optional[int] = 0,
                 underlying: Optional[str] = None,
                 last_date: Optional[datetime.datetime] = None):
        self._ticker = ticker
        self._name = name
        self._list_level = list_level
        self._underlying = underlying
        self._last_date = last_date

    def get_ticker(self) -> str:
        return self._ticker

    def get_name(self) -> str:
        return self._name

    def get_list_level(self) -> int:
        return self._list_level

    def get_underlying(self) -> Optional[str]:
        return self._underlying if self._underlying is not None else self.get_ticker()

    def get_last_date(self) -> Optional[datetime.datetime]:
        return self._last_date

    def get_last_date_str(self) -> Optional[str]:
        return self._last_date.strftime("%Y-%m-%d") if self._last_date is not None else None

    def __repr__(self) -> str:
        return f"ISSSecurity({self.get_ticker()}, {self.get_name()})"


class Trend(Enum):
    UP = "Up"
    DOWN = "Down"
    CHG_UP = "Changing to Up"
    CHG_DOWN = "Changing to Down"
    CROSS = "?"


class AssetState:
    def __init__(self,
                 security: ISSSecurity,
                 tf: str,
                 status: bool,
                 trend: Trend,
                 image: Image,
                 interest: bool,
                 potential: Optional[float] = None,
                 potential_trend: Optional[Trend] = None,
                 closestLevel: Optional[str] = 'none'):
        self._security = security
        self._tf = tf
        self._status = status
        self._trend = trend
        self._image = image
        self._interest = interest
        self._potential = potential
        self._file_path = None
        self._potential_trend = potential_trend
        self._closestLevel = closestLevel

    def get_security(self) -> ISSSecurity:
        return self._security

    def get_status(self) -> bool:
        return self._status

    def get_tf(self) -> str:
        return self._tf

    def get_trend_value(self) -> Trend:
        return self._trend

    def get_potential_trend_value(self) -> Optional[Trend]:
        return self._potential_trend

    def get_closestLevel(self):
        return self._closestLevel

    def get_trend(self) -> str:
        return self._trend.name if self._trend is not None else "-"

    def get_image(self) -> Image:
        return self._image

    def get_image_width(self) -> int:
        if self._image is None:
            return 0
        h, w = self._image.size
        return w

    def get_image_height(self) -> int:
        if self._image is None:
            return 0
        h, w = self._image.size
        return h

    def get_image_path(self, parent_dir: str) -> str:
        if self._image is not None:
            if self._file_path is None:
                ticker = self._security.get_underlying()
                tf = self._tf
                pic_dir = os.path.join(parent_dir, "pic")
                if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
                self._file_path = os.path.abspath(os.path.join(pic_dir, f"{ticker}-{tf}.png"))
                self._image.save(self._file_path)
            return self._file_path
        return ""

    def save_image(self, parent_dir: str):
        if self._image is not None:
            ticker = self._security.get_underlying()
            tf = self._tf
            pic_dir = os.path.join(parent_dir, "pic")
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            __file_path = os.path.abspath(os.path.join(pic_dir, f"{ticker}-{tf}.png"))
            self._image.save(__file_path)

    def get_interest(self) -> bool:
        return self._interest

    def get_potential_str(self) -> str:
        return f"{self._potential:.2f}%" if not math.isclose(self._potential, 0) else "-"

    def get_potential(self) -> float:
        return self._potential

    def get_style(self) -> str:
        if self._interest:
            return "green"
        if not math.isclose(self._potential, 0.0) and self._potential > 0.05:
            if isUpTrend(self._trend):
                return "blue"
            if isDownTrend(self._trend):
                return "red"
        return ""

    def updateInterest(self, new_interest: bool):
        self._interest = new_interest

    def __lt__(self, other):
        return self._potential < other.get_potential()

    def __repr__(self):
        return f"AssetState(tf={self._tf}, status={self._status}, interest={self._interest}, potential={self.get_potential_str()})"


class ScannerResult:
    def __init__(self,
                 security: ISSSecurity,
                 last_price: Optional[float],
                 states: list[AssetState]):
        self._security = security
        self._last_price = last_price
        self._states = states

    def get_security(self) -> ISSSecurity:
        return self._security

    def hasValidState(self) -> bool:
        _sts = self.get_states()
        _res = False
        for _s in _sts:
            if _s.get_status():
                _res = True
                break
        return _res

    def hasAllValidStates(self) -> bool:
        _sts = self.get_states()
        _res = True
        for _s in _sts:
            if not _s.get_status():
                _res = False
                break
        return _res

    def hasInterest(self) -> bool:
        _sts = self.get_states()
        _res = False
        for _s in _sts:
            if _s.get_interest():
                _res = True
                break
        return _res

    def get_states(self) -> list[AssetState]:
        return self._states

    def get_max_potential(self) -> float:
        return max(b.get_potential() for b in self.get_states())

    def get_last_price(self) -> str:
        return "-" if self._last_price is None else "{:.2f}".format(self._last_price)

    def saveImages(self, parent_dir: str):
        for state in self._states:
            state.save_image(parent_dir)

    def __repr__(self) -> str:
        return f"ScannerResult(symbol={self.get_security().get_ticker()}, lastPrice={self.get_last_price()})"


async def get_moex_securities(session: aiohttp.ClientSession) -> List[ISSSecurity]:
    result = []
    url = aiomoex.request_helpers.make_url(engine="stock",
                                           market="shares",
                                           ending="securities")
    query = aiomoex.request_helpers.make_query()
    query["sectypes"] = "1"
    data = await aiomoex.request_helpers.get_short_data(session, url, "securities", query)
    for line in data:
        if line['BOARDID'] == "TQBR":
            ticker = line['SECID']
            name = line['SECNAME']
            list_level = line['LISTLEVEL']
            result.append(ISSSecurity(ticker, name, list_level=list_level))
            print(f"{ticker}: {name} - {list_level}")
    return result


async def get_moex_futures_securities(session: aiohttp.ClientSession,
                                      _filter: list[str]) -> List[ISSSecurity]:
    result = []
    url = aiomoex.request_helpers.make_url(engine="futures",
                                           market="forts",
                                           ending="securities")
    query = aiomoex.request_helpers.make_query()
    data = await aiomoex.request_helpers.get_short_data(session, url, "securities", query)
    now_date = datetime.datetime.now()
    map_asset_futures = {}

    for line in data:
        if line['BOARDID'] == "RFUD":
            ticker = line['SECID']
            name = line['SECNAME']
            last_date_str = line['LASTTRADEDATE']
            last_date = pd.to_datetime(last_date_str, format='%Y-%m-%d')
            underlying = line["ASSETCODE"]
            if last_date > now_date and underlying in _filter:
                underlying_futures = map_asset_futures.get(underlying, [])
                underlying_futures.append(ISSSecurity(ticker, name, underlying=underlying, last_date=last_date))
                map_asset_futures[underlying] = underlying_futures

    for und, futs in map_asset_futures.items():
        sorted_res = sorted(futs,
                            key=lambda a: (
                                a.get_last_date()
                            ),
                            reverse=False)
        current_futures = sorted_res[0]
        result.append(current_futures)
        print(f"{current_futures.get_ticker()}: {current_futures.get_name()} - {current_futures.get_last_date_str()}")
    return result


def get_trend(vwma_fast: float, vwma_slow: float) -> Trend:
    if math.isclose(vwma_fast, vwma_slow):
        return Trend.CROSS
    if vwma_fast > vwma_slow:
        return Trend.UP
    else:
        return Trend.DOWN


def get_change_trend(trend: Trend) -> Trend:
    if trend == Trend.UP:
        return Trend.CHG_UP
    if trend == Trend.DOWN:
        return Trend.CHG_DOWN
    else:
        return trend


def isUpTrend(trend: Trend) -> bool:
    return trend == Trend.UP or trend == Trend.CHG_UP


def isDownTrend(trend: Trend) -> bool:
    return trend == Trend.DOWN or trend == Trend.CHG_DOWN


def getClosestLevel(dataset: pd.DataFrame):
    #TODO: Check - maybe we do not need to check prev candle in case monitoring will be constant
    #TODO: Check low/high separately and provide closest direction - UP/DOWN
    _hasPrevClose = True
    prev_close = 0.0
    prev_high = 0.0
    prev_low = 0.0
    if len(dataset) < 2:
        _hasPrevClose = False
    else:
        prev_close = dataset["close"].iloc[-2]
        prev_high = dataset["high"].iloc[-2]
        prev_low = dataset["low"].iloc[-2]
    last_high = dataset["high"].iloc[-1]
    last_low = dataset["low"].iloc[-1]
    last_close = dataset["close"].iloc[-1]
    columns = []
    if 'vwma_fast' in dataset.columns:
        columns.append('vwma_fast')
    if 'vwma_slow' in dataset.columns:
        columns.append('vwma_slow')
    if 'ema_fast' in dataset.columns:
        columns.append('ema_fast')
    if 'ema_slow' in dataset.columns:
        columns.append('ema_slow')
    dist = [(x, abs(last_close - dataset[x].iloc[-1])/last_close) for x in columns]
    lev_min = min(dist, key=lambda x: x[1])
    current_level = lev_min[0] if lev_min[1] < 0.0005 else ''
    #check current and previous levels
    if current_level == '' and _hasPrevClose:
        dist = [(x, abs(prev_close - dataset[x].iloc[-2]) / prev_close) for x in columns]
        lev_min = min(dist, key=lambda x: x[1])
        current_level = lev_min[0] if lev_min[1] < 0.0005 else ''
    if current_level == '':
        dist = [(x, (last_high - dataset[x].iloc[-1])*(last_low - dataset[x].iloc[-1])) for x in columns]
        lev_min = min(dist, key=lambda x: x[1])
        current_level = lev_min[0] if lev_min[1] <= 0.0 else ''
    if current_level == '' and _hasPrevClose:
        dist = [(x, (prev_high - dataset[x].iloc[-2])*(prev_low - dataset[x].iloc[-2])) for x in columns]
        lev_min = min(dist, key=lambda x: x[1])
        current_level = lev_min[0] if lev_min[1] <= 0.0 else ''
    return current_level


def get_potential(dataset: pd.DataFrame, checkDownTrend: Optional[bool] = False):
    count = 0
    cur_trend = None
    reversed_dataset = dataset.iloc[::-1]
    for date, close, fast, slow in zip(reversed_dataset["datetime"], reversed_dataset["close"], reversed_dataset["vwma_fast"],
                                       reversed_dataset["vwma_slow"]):
        trend = get_trend(fast, slow)
        if not cur_trend:
            cur_trend = trend
            continue
        if trend != cur_trend:
            cur_trend = cur_trend if count > 20 else get_change_trend(cur_trend)
            break

    interest = False
    potential = 0.0
    last_row = dataset.tail(1)
    last_close = last_row["close"].tolist()[0]
    last_low = last_row["low"].tolist()[0]
    last_vwma_fast = last_row["vwma_fast"].tolist()[0]
    last_vwma_slow = last_row["vwma_slow"].tolist()[0]

    vwma_fast_trend = None
    vwma_slow_trend = None
    if len(dataset) > 2:
        vwma_fast_past = dataset["vwma_fast"].iloc[-3]
        vwma_fast_last = dataset["vwma_fast"].iloc[-1]
        vwma_slow_past = dataset["vwma_fast"].iloc[-3]
        vwma_slow_last = dataset["vwma_fast"].iloc[-1]
        if vwma_slow_last - vwma_slow_past > 0:
            vwma_slow_trend = Trend.UP
        else:
            vwma_slow_trend = Trend.DOWN
        if vwma_fast_last - vwma_fast_past > 0:
            vwma_fast_trend = Trend.UP
        else:
            vwma_fast_trend = Trend.DOWN

    if isUpTrend(cur_trend) and last_vwma_fast > last_close:
        potential = (last_vwma_fast - last_close) / last_close * 100
    if checkDownTrend and isDownTrend(cur_trend) and last_vwma_fast < last_close:
        potential = (last_close - last_vwma_fast) / last_close * 100
    if last_vwma_fast > last_vwma_slow >= last_close and (isUpTrend(vwma_fast_trend) or isUpTrend(vwma_slow_trend)):
        interest = True
    if checkDownTrend and last_vwma_fast < last_vwma_slow < last_close and (isDownTrend(vwma_fast_trend) or isDownTrend(vwma_slow_trend)):
        interest = True

    closestLevel = getClosestLevel(dataset)
    return cur_trend, interest, potential, closestLevel


def get_potential_trend_change(dataset: pd.DataFrame):
    if dataset.empty or len(dataset) < 2:
        return None
    last_rows = dataset.tail(2)
    last_vwma_fast = last_rows["vwma_fast"].tolist()
    last_vwma_slow = last_rows["vwma_slow"].tolist()
    delta_fast = last_vwma_fast[1] - last_vwma_fast[0]
    delta_slow = last_vwma_slow[1] - last_vwma_slow[0]
    if delta_fast > 0 and delta_fast * delta_slow > 0:
        return Trend.UP
    if delta_fast < 0 < delta_fast * delta_slow:
        return Trend.DOWN

    return Trend.CROSS


def get_state(security: ISSSecurity, tf: str, data_vwma: pd.DataFrame, checkDownTrend: Optional[bool] = False) -> AssetState:
    dataset = data_vwma.tail(256)
    has_data = not data_vwma.empty and len(data_vwma) > 1
    trend = None
    interest = False
    potential = 0.0
    potential_trend = None
    img = None
    closestLevel = 'none'
    if has_data:
        trend, interest, potential, closestLevel = get_potential(dataset, checkDownTrend)
        potential_trend = get_potential_trend_change(dataset)
        img = functions.create_image(dataset, True)
    return AssetState(security, tf, has_data, trend, img, interest, potential,
                      potential_trend=potential_trend,
                      closestLevel=closestLevel)


async def get_stock_security_state(session: aiohttp.ClientSession, security: ISSSecurity) -> ScannerResult:
    # 2 month for 1H ( 6 to be converted to 4H)
    from_h1_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    # 1 year for 1D
    from_d1_date = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime("%Y-%m-%d")
    # 8 years for 1W
    from_w1_date = (datetime.datetime.now() - datetime.timedelta(days=365*8)).strftime("%Y-%m-%d")
    data_h1 = await functions.get_stock_candles(session, security.get_ticker(), "H1", from_h1_date, None,
                                                file_store=os.path.join("_scan", "csv", f"stock-{security.get_ticker()}_H1.csv"))
    data_d1 = await functions.get_stock_candles(session, security.get_ticker(), "D1", from_d1_date, None,
                                                file_store=os.path.join("_scan", "csv",
                                                                        f"stock-{security.get_ticker()}_D1.csv"))
    data_w1 = await functions.get_stock_candles(session, security.get_ticker(), "W1", from_w1_date, None,
                                                file_store=os.path.join("_scan", "csv", f"stock-{security.get_ticker()}_W1.csv"))
    print(f"Received data for {security.get_ticker()} H1-{'OK' if not data_h1.empty else 'NOK'} D1-{'OK' if not data_d1.empty else 'NOK'} W1-{'OK' if not data_w1.empty else 'NOK'}")
    data_h4 = pd.DataFrame()
    last_price = None
    if not data_h1.empty:
        last_price = data_h1.tail(1)["close"].tolist()[0]
        data_h4 = functions.aggregate(data_h1, 240)
    data_h1_vwma = pd.DataFrame()
    data_h4_vwma = pd.DataFrame()
    data_d1_vwma = pd.DataFrame()
    data_w1_vwma = pd.DataFrame()
    if not data_h1.empty:
        data_h1_vwma = functions.get_vwma(data_h1, 50, 100, 200, 100, 200, drop_nan=False)
    if not data_h4.empty:
        data_h4_vwma = functions.get_vwma(data_h4, 50, 100, 200, 100, 200, drop_nan=False)
    if not data_d1.empty:
        data_d1_vwma = functions.get_vwma(data_d1, 50, 100, 200, 100, 200, drop_nan=False)
    if not data_w1.empty:
        data_w1_vwma = functions.get_vwma(data_w1, 50, 100, 200, 100, 200, drop_nan=False)

    return ScannerResult(security,
                         last_price,
                         [
                             get_state(security, "H1", data_h1_vwma),
                             get_state(security, "H4", data_h4_vwma),
                             get_state(security, "D1", data_d1_vwma),
                             get_state(security, "W1", data_w1_vwma)
                         ])


def isSameTrend(asset1: AssetState, asset2: AssetState, check_potential1: bool = False, check_potential2: bool = False):
    trend1 = asset1.get_potential_trend_value() if check_potential1 else asset1.get_trend_value()
    trend2 = asset2.get_potential_trend_value() if check_potential2 else asset2.get_trend_value()
    if trend1 is None or trend2 is None:
        return False
    if trend1 == Trend.CROSS or trend2 == Trend.CROSS:
        return False
    isUp1 = trend1 == Trend.UP or trend1 == Trend.CHG_UP
    isUp2 = trend2 == Trend.UP or trend2 == Trend.CHG_UP
    return isUp1 == isUp2


def update_interest(states: list[AssetState], check_next_only: bool = False):
    if len(states) < 2:
        return
    #Remove interest on the highest state - it should be just a reference
    states[len(states) - 1].updateInterest(False)

    # TODO: revise this logic
    # Remove interest on the lowest state - it should be just a reference
    if states[0].get_interest():
        _dropInterest = True
        for _idx in range(1, len(states)):
            if states[_idx].get_closestLevel() != '':
                _dropInterest = False
        if _dropInterest:
            states[0].updateInterest(False)

    for _idx in reversed(range(0, len(states) - 1)):
        _state = states[_idx]
        if _state.get_interest():
            if check_next_only:
                _next_state = states[_idx+1]
                _first_state = states[0]
                if not (isSameTrend(_state, _next_state) and isSameTrend(_state, _first_state, check_potential2=True)):
                    _state.updateInterest(False)
            else:
                _same_higher_trend = True
                for _back_idx in range(_idx + 1, len(states)):
                    _highState = states[_back_idx]
                    if not isSameTrend(_state, _highState):
                        _same_higher_trend = False
                        break
                if not _same_higher_trend:
                    # since upper trends is different - it's not a correction at the moment
                    _state.updateInterest(False)
    res = [{s.get_tf(): s.get_interest()} for s in states]
    print(f"{states[0].get_security().get_underlying()} - {res}")


async def get_futures_security_state(session: aiohttp.ClientSession, security: ISSSecurity) -> ScannerResult:
    # M1 - for last 5 days
    from_m1_date = (datetime.datetime.now() - datetime.timedelta(days=25)).strftime("%Y-%m-%d")
    from_h1_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
    data_m1 = await functions.get_futures_candles(session, security.get_ticker(), "M1", from_m1_date, None,
                                                  file_store=os.path.join("_scan", "csv", f"futures-{security.get_underlying()}_M1.csv"))
    data_h1 = await functions.get_futures_candles(session, security.get_ticker(), "H1", from_h1_date, None,
                                                  file_store=os.path.join("_scan", "csv", f"futures-{security.get_underlying()}_H1.csv"))

    print(f"Received data for {security.get_ticker()}")
    data_m5 = pd.DataFrame()
    data_m10 = pd.DataFrame()
    data_m15 = pd.DataFrame()
    data_m30 = pd.DataFrame()
    data_h3 = pd.DataFrame()
    last_price = None
    if not data_m1.empty:
        last_price = data_m1.tail(1)["close"].tolist()[0]
        data_m5 = functions.aggregate(data_m1, 5)
        data_m10 = functions.aggregate(data_m1, 10)
        data_m15 = functions.aggregate(data_m1, 15)
        data_m30 = functions.aggregate(data_m1, 30)
    if not data_h1.empty:
        data_h3 = functions.aggregate(data_h1, 180)
    data_m1_vwma = pd.DataFrame()
    data_m5_vwma = pd.DataFrame()
    data_m10_vwma = pd.DataFrame()
    data_m15_vwma = pd.DataFrame()
    data_m30_vwma = pd.DataFrame()
    data_h1_vwma = pd.DataFrame()
    data_h3_vwma = pd.DataFrame()
    if not data_m1.empty:
        data_m1_vwma = functions.get_vwma(data_m1, 50, 100, 200, 100, 200, drop_nan=False)
        data_m5_vwma = functions.get_vwma(data_m5, 50, 100, 200, 100, 200, drop_nan=False)
        data_m10_vwma = functions.get_vwma(data_m10, 50, 100, 200, 100, 200, drop_nan=False)
        data_m15_vwma = functions.get_vwma(data_m15, 50, 100, 200, 100, 200, drop_nan=False)
        data_m30_vwma = functions.get_vwma(data_m30, 50, 100, 200, 100, 200, drop_nan=False)
    if not data_h1.empty:
        data_h1_vwma = functions.get_vwma(data_h1, 50, 100, 200, 100, 200, drop_nan=False)
        data_h3_vwma = functions.get_vwma(data_h3, 50, 100, 200, 100, 200, drop_nan=False)
    _state_m1 = get_state(security, "M1", data_m1_vwma, checkDownTrend=True)
    _state_m5 = get_state(security, "M5", data_m5_vwma, checkDownTrend=True)
    _state_m10 = get_state(security, "M10", data_m10_vwma, checkDownTrend=True)
    _state_m15 = get_state(security, "M15", data_m15_vwma, checkDownTrend=True)
    _state_m30 = get_state(security, "M30", data_m30_vwma, checkDownTrend=True)
    _state_h1 = get_state(security, "H1", data_h1_vwma, checkDownTrend=True)
    _state_h3 = get_state(security, "H3", data_h3_vwma, checkDownTrend=True)
    _states = [_state_m1, _state_m5, _state_m10, _state_m15, _state_m30, _state_h1, _state_h3]
    update_interest(_states, check_next_only=True)
    _result = ScannerResult(security, last_price, _states)
    parent_dir = os.path.join("_scan", "csv", "futures")
    _result.saveImages(parent_dir)
    return _result


def notifyResultsWithBot(all_results: list[ScannerResult], parent_dir: str):
    interesting_results = [x for x in all_results if x.hasAllValidStates() and x.hasInterest()]
    if len(interesting_results) == 0:
        return

    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    OWNER_TELE_ID = int(os.environ.get('OWNER_TELE_ID'))
    if BOT_TOKEN is None or len(BOT_TOKEN) == 0:
        print(f"Token not defined")
        return

    bot = telebot.TeleBot(BOT_TOKEN)
    if len(interesting_results) > 0:
        scan_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        bot.send_message(OWNER_TELE_ID, f"Scan results on {scan_date}: {len(interesting_results)} found:")
    for result in interesting_results:
        _security = result.get_security()
        _last_price = result.get_last_price()
        _medias = []
        _str = f"{_security.get_name()}: price:{_last_price}"
        for state in result.get_states():
            _tf = state.get_tf()
            _potential = state.get_potential_str()
            _closest_level = state.get_closestLevel()
            _msg = f"{_security.get_ticker()}-{_security.get_name()} Price: {_last_price} Potential-{_tf}: {_potential} ClosestLevel: {_closest_level}"
            _img_path = state.get_image_path(parent_dir)
            if state.get_interest() or state.get_potential() > 0.0 or _closest_level != '':
                _str = _str + f" {_tf}:{_potential}:{_closest_level}"
            if _img_path:
                _medias.append(
                    telebot.types.InputMediaPhoto(open(_img_path, 'rb'), caption=_msg)
                )
        #send text overview
        bot.send_message(OWNER_TELE_ID, _str)
        if len(_medias) > 0:
            # send charts overview
            bot.send_media_group(OWNER_TELE_ID, _medias)
    #stop bot
    bot.stop_bot()


def print_scanner_results(description: str, results: List[ScannerResult], isFutures: bool = False):
    scan_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    results_count = len(results)
    parent_dir = os.path.join("_scan", "out_" + description)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    filtered_results = [x for x in results if x.hasAllValidStates()]
    total_with_data_count = len(filtered_results)
    interesting_results = [x for x in filtered_results if x.hasInterest() or x.get_max_potential() > 5]
    interesting_results_count = len(interesting_results)

    # print out all futures if there's nothing interesting found
    reportable_results = filtered_results if isFutures and len(interesting_results) == 0 else interesting_results

    # Sorting max(potential) -> list level -> ticker
    reportable_results = sorted(reportable_results,
                                key=lambda a: (
                                    a.get_max_potential(),
                                    a.get_security().get_list_level(),
                                    a.get_security().get_ticker()
                                ),
                                reverse=True)

    res_params = []
    tfs = ['H1', 'H4', 'D1', 'W1']
    if len(reportable_results) > 0:
        _states = reportable_results[0].get_states()
        tfs = [x.get_tf() for x in _states]
    for res in reportable_results:
        res_params.append({
            'name': res.get_security().get_underlying(),
            'description': res.get_security().get_name(),
            'last_price': res.get_last_price(),
            'list_level': res.get_security().get_list_level(),
            'states': [{'trend': x.get_trend(),
                        'style': x.get_style(),
                        'tf': x.get_tf(),
                        'potential': x.get_potential_str(),
                        'closestLevel': x.get_closestLevel(),
                        'img': x.get_image_path(parent_dir)} for x in res.get_states()],
        })

    with open('scanner_template.tmpl', 'r') as file:
        template = file.read()
    data = {
        'description': description,
        'total_scanned': results_count,
        'total_with_data': total_with_data_count,
        'interesting_results': interesting_results_count,
        'tfs': [{'tf': tf} for tf in tfs],
        'scan_list': res_params,
        'scan_date': scan_date
    }
    output = chevron.render(template=template, data=data)
    out_file_path = os.path.join(parent_dir, f"index.html")
    out_file = open(out_file_path, "w", encoding="utf-8")
    out_file.write(output)
    out_file.close()

    jsonString = json.dumps(data)
    out_json_path = os.path.join(parent_dir, f"data.json")
    jsonFile = open(out_json_path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(f"Found {interesting_results_count} interesting results\n")
    print(f"Result is written to {out_file_path}")

    if isFutures:
        notifyResultsWithBot(results, parent_dir)


async def stock_screen():
    connector = aiohttp.TCPConnector(force_close=False, use_dns_cache=False, limit=1, limit_per_host=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        securities = await get_moex_securities(session)
        print(f"Found {len(securities)} tickers on MOEX")
        tasks = []
        for security in securities:
            tasks.append(asyncio.create_task(get_stock_security_state(session, security)))
        results = await asyncio.gather(*tasks)
        print_scanner_results("Stocks", results)
        end = time.time()
        total_time = end - start
        print(f"Execution completed in {str(total_time)} sec")


async def futures_screen():
    connector = aiohttp.TCPConnector(force_close=True, limit=5, limit_per_host=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        #securities = await get_moex_futures_securities(session, ["Si", "BR", "GOLD", "NG", "MIX", "RTS", "SILV"])
        securities = await get_moex_futures_securities(session, ["Si", "BR", "GOLD", "NG"])
        print(f"Found {len(securities)} tickers on MOEX")
        tasks = []
        for security in securities:
            tasks.append(asyncio.create_task(get_futures_security_state(session, security)))
        results = await asyncio.gather(*tasks)
        print_scanner_results("Futures", results, True)
        end = time.time()
        total_time = end - start
        print(f"Execution completed in {str(total_time)} sec")


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Terminate script"
    print(msg, end="", flush=True)
    exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)

    # make script location to become working directory
    abspath = os.path.abspath(__file__)
    dir_name = os.path.dirname(abspath)
    os.chdir(dir_name)

    loop = asyncio.get_event_loop()  # создаем цикл
    n = len(sys.argv)
    if n > 1 and sys.argv[1] == 'futures':
        task = loop.create_task(  # в цикл добавляем 1 задачу
            futures_screen()
        )
    else:
        task = loop.create_task(  # в цикл добавляем 1 задачу
            stock_screen()
        )
    loop.run_until_complete(task)  # ждем окончания выполнения цикла
