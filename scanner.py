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
                 underlying: Optional[str] = "",
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
        return self._underlying

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
                 potential: Optional[float] = None):
        self._security = security
        self._tf = tf
        self._status = status
        self._trend = trend
        self._image = image
        self._interest = interest
        self._potential = potential
        self._file_path = None

    def get_status(self) -> bool:
        return self._status

    def get_tf(self) -> str:
        return self._tf

    def get_trend_value(self) -> Trend:
        return self._trend

    def get_trend(self) -> str:
        return self._trend.name if self._trend is not None else "-"

    def get_image(self) -> Image:
        return self._image

    def get_image_path(self, parent_dir: str) -> str:
        if self._image is not None:
            if self._file_path is None:
                ticker = self._security.get_ticker()
                tf = self._tf
                pic_dir = os.path.join(parent_dir, "pic")
                if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
                self._file_path = os.path.abspath(os.path.join(pic_dir, f"{ticker}-{tf}.png"))
                self._image.save(self._file_path)
            return self._file_path
        return ""

    def get_interest(self) -> bool:
        return self._interest

    def get_potential_str(self) -> str:
        return f"{self._potential:.2f}%" if not math.isclose(self._potential, 0) else "-"

    def get_potential(self) -> float:
        return self._potential

    def get_style(self) -> str:
        return "" if math.isclose(self._potential, 0.0) else "green"

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

    def get_last_price(self) -> str:
        return "-" if self._last_price is None else "{:.2f}".format(self._last_price)

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


def get_potential(dataset: pd.DataFrame, futures: bool = False):
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
    if futures:
        if last_vwma_fast > last_vwma_slow and last_close < last_vwma_fast:
            interest = True
            potential = (last_vwma_fast - last_close) / last_close * 100
    else:
        if last_vwma_fast > last_vwma_slow >= last_close:
            interest = True
            potential = (last_vwma_fast - last_close) / last_close * 100
    return cur_trend, interest, potential


def get_state(security: ISSSecurity, tf: str, data_vwma: pd.DataFrame, futures: bool = False) -> AssetState:
    dataset = data_vwma.tail(256)
    has_data = not data_vwma.empty
    trend = None
    interest = False
    potential = 0.0
    img = None
    if has_data:
        trend, interest, potential = get_potential(dataset, futures)
        img = functions.create_image(dataset, True)
    return AssetState(security, tf, has_data, trend, img, interest, potential)


async def get_stock_security_state(session: aiohttp.ClientSession, security: ISSSecurity) -> ScannerResult:
    # 2 month for 1H ( 6 to be converted to 4H)
    from_h1_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    # 1 year for 1D
    from_d1_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    # 8 years for 1W
    from_w1_date = (datetime.datetime.now() - datetime.timedelta(days=365*8)).strftime("%Y-%m-%d")
    data_d1 = await functions.get_stock_candles(session, security.get_ticker(), "D1", from_d1_date, None,
                                                file_store=os.path.join("_scan", "csv", f"stock-{security.get_ticker()}_D1.csv"))
    data_h1 = await functions.get_stock_candles(session, security.get_ticker(), "H1", from_h1_date, None,
                                                file_store=os.path.join("_scan", "csv", f"stock-{security.get_ticker()}_H1.csv"))
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
        data_h1_vwma = functions.get_vwma(data_h1, 50, 100, 200, drop_nan=False)
    if not data_h4.empty:
        data_h4_vwma = functions.get_vwma(data_h4, 50, 100, 200, drop_nan=False)
    if not data_d1.empty:
        data_d1_vwma = functions.get_vwma(data_d1, 50, 100, 200, drop_nan=False)
    if not data_w1.empty:
        data_w1_vwma = functions.get_vwma(data_w1, 50, 100, 200, drop_nan=False)

    return ScannerResult(security,
                         last_price,
                         [
                             get_state(security, "H1", data_h1_vwma),
                             get_state(security, "H4", data_h4_vwma),
                             get_state(security, "D1", data_d1_vwma),
                             get_state(security, "W1", data_w1_vwma)
                         ])


def isSameTrend(trend1: Trend, trend2: Trend):
    if trend1 is None or trend2 is None:
        return False
    isUp1 = trend1 == Trend.UP or trend1 == Trend.CHG_UP
    isUp2 = trend2 == Trend.UP or trend2 == Trend.CHG_UP
    return isUp1 == isUp2


def update_interest(states: list[AssetState], check_next_only: bool):
    if len(states) < 2:
        return
    for _idx in reversed(range(0, len(states) - 1)):
        _state = states[_idx]
        if _state.get_interest():
            _trend = _state.get_trend()
            if check_next_only:
                _next_state = states[_idx+1]
                if not isSameTrend(_state.get_trend_value(), _next_state.get_trend_value()):
                    _state.updateInterest(False)
            else:
                _same_higher_trend = True
                for _back_idx in range(_idx + 1, len(states)):
                    _highState = states[_back_idx]
                    if not isSameTrend(_state.get_trend_value(), _highState.get_trend_value()):
                        _same_higher_trend = False
                        break
                if not _same_higher_trend:
                    # since upper trends is different - it's not a correction at the moment
                    _state.updateInterest(False)


async def get_futures_security_state(session: aiohttp.ClientSession, security: ISSSecurity) -> ScannerResult:
    # M1 - for last 5 days
    from_m1_date = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    from_m10_date = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    from_h1_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    data_m1 = await functions.get_futures_candles(session, security.get_ticker(), "M1", from_m1_date, None)
    data_m10 = await functions.get_futures_candles(session, security.get_ticker(), "M10", from_m10_date, None)
    data_h1 = await functions.get_futures_candles(session, security.get_ticker(), "H1", from_h1_date, None)

    print(f"Received data for {security.get_ticker()}")
    data_m5 = pd.DataFrame()
    data_m15 = pd.DataFrame()
    data_m30 = pd.DataFrame()
    last_price = None
    if not data_m1.empty:
        last_price = data_m1.tail(1)["close"].tolist()[0]
        data_m5 = functions.aggregate(data_m1, 5)
        data_m15 = functions.aggregate(data_m1, 15)
    if not data_m10.empty:
        data_m30 = functions.aggregate(data_m10, 30)
    data_m1_vwma = pd.DataFrame()
    data_m5_vwma = pd.DataFrame()
    data_m15_vwma = pd.DataFrame()
    data_m30_vwma = pd.DataFrame()
    data_h1_vwma = pd.DataFrame()
    if not data_m1.empty:
        data_m1_vwma = functions.get_vwma(data_m1, 50, 100, 200, drop_nan=False)
        data_m5_vwma = functions.get_vwma(data_m5, 50, 100, 200, drop_nan=False)
        data_m15_vwma = functions.get_vwma(data_m15, 50, 100, 200, drop_nan=False)
    if not data_m30.empty:
        data_m30_vwma = functions.get_vwma(data_m30, 50, 100, 200, drop_nan=False)
    if not data_h1.empty:
        data_h1_vwma = functions.get_vwma(data_h1, 50, 100, 200, drop_nan=False)
    _state_m1 = get_state(security, "M1", data_m1_vwma, False)
    _state_m5 = get_state(security, "M5", data_m5_vwma, False)
    _state_m15 = get_state(security, "M15", data_m15_vwma, False)
    _state_m30 = get_state(security, "M30", data_m30_vwma, False)
    _state_h1 = get_state(security, "H1", data_h1_vwma, False)
    update_interest([_state_m1, _state_m5, _state_m15, _state_h1], check_next_only=True)
    return ScannerResult(security,
                         last_price,
                         [_state_m1, _state_m5, _state_m15, _state_m30, _state_h1])


def notifyResultsWithBot(interesting_results: list[ScannerResult], parent_dir: str):
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
            _msg = f"{_security.get_ticker()}-{_security.get_name()} Price: {_last_price} Potential-{_tf}: {_potential}"
            _img_path = state.get_image_path(parent_dir)
            if state.get_interest():
                _str = _str + f" {_tf}:{_potential}"
            if _img_path:
                _medias.append(
                    telebot.types.InputMediaPhoto(open(_img_path, 'rb'), caption=_msg)
                )
        #send text overview
        bot.send_message(OWNER_TELE_ID, _str)
        if len(_medias) > 0:
            # send charts overview
            bot.send_media_group(OWNER_TELE_ID, _medias)

    bot.stop_bot()


def print_scanner_results(description: str, results: List[ScannerResult], isFutures: bool = False):
    scan_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    results_count = len(results)
    parent_dir = os.path.join("_scan", "out_" + description)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    filtered_results = [x for x in results if x.hasAllValidStates()]
    total_with_data_count = len(filtered_results)
    interesting_results = [x for x in filtered_results if x.hasInterest()]
    interesting_results_count = len(interesting_results)
    # Sorting max(potential) -> list level -> ticker
    filtered_results = sorted(interesting_results,
                              key=lambda a: (
                                  max(b.get_potential() for b in a.get_states()),
                                  a.get_security().get_list_level(),
                                  a.get_security().get_ticker()
                              ),
                              reverse=True)

    res_params = []
    tfs = ['H1', 'H4', 'D1', 'W1']
    if len(filtered_results) > 0:
        _states = filtered_results[0].get_states()
        tfs = [x.get_tf() for x in _states]
    for res in filtered_results:
        res_params.append({
            'name': res.get_security().get_ticker(),
            'description': res.get_security().get_name(),
            'last_price': res.get_last_price(),
            'list_level': res.get_security().get_list_level(),
            'states': [{'trend': x.get_trend(),
                        'style': x.get_style(),
                        'tf': x.get_tf(),
                        'potential': x.get_potential_str(),
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
    n = out_file.write(output)
    out_file.close()

    jsonString = json.dumps(data)
    out_json_path = os.path.join(parent_dir, f"data.json")
    jsonFile = open(out_json_path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(f"Found {interesting_results_count} interesting results\n")
    print(f"Result is written to {out_file_path}")

    if isFutures:
        notifyResultsWithBot(interesting_results, parent_dir)


async def stock_screen():
    connector = aiohttp.TCPConnector(force_close=True, limit=5, limit_per_host=5)
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
        securities = await get_moex_futures_securities(session, ["Si", "BR", "GOLD", "NG", "MIX", "RTS", "SILV"])
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
