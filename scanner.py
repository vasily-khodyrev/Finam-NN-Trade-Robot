import asyncio
import datetime
import math
import os
import time
from enum import Enum
from typing import Optional, List

import aiohttp
import aiomoex
import chevron
import pandas as pd
from PIL import Image

import functions


class ISSSecurity:
    def __init__(self, ticker: str, name: str, list_level: int):
        self._ticker = ticker
        self._name = name
        self._list_level = list_level

    def get_ticker(self) -> str:
        return self._ticker

    def get_name(self) -> str:
        return self._name

    def get_list_level(self) -> int:
        return self._list_level


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

    def get_trend(self) -> str:
        return self._trend.name

    def get_image(self) -> Image:
        return self._image

    def get_image_path(self, parent_dir: str) -> str:
        if self._image is not None:
            if self._file_path is None:
                ticker = self._security.get_ticker()
                tf = self._tf
                pic_dir = os.path.join(parent_dir, "pic")
                if not os.path.exists(pic_dir): os.makedirs(pic_dir)
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


class ScannerResult:
    def __init__(self, security: ISSSecurity, h1_state: AssetState, h4_state: AssetState, d1_state: AssetState):
        self._security = security
        self._h1_state = h1_state
        self._h4_state = h4_state
        self._d1_state = d1_state

    def get_security(self) -> ISSSecurity:
        return self._security

    def get_h1_state(self) -> AssetState:
        return self._h1_state

    def get_h4_state(self) -> AssetState:
        return self._h4_state

    def get_d1_state(self) -> AssetState:
        return self._d1_state


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
            result.append(ISSSecurity(ticker, name, list_level))
            print(f"{ticker}: {name} - {list_level}")
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


def get_potential(dataset: pd.DataFrame):
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
    last_vwma_fast = last_row["vwma_fast"].tolist()[0]
    last_vwma_slow = last_row["vwma_slow"].tolist()[0]
    if last_vwma_fast > last_vwma_slow >= last_close:
        interest = True
        potential = (last_vwma_fast - last_close) / last_close * 100
    if last_close >= last_vwma_slow > last_vwma_fast:
        interest = True
        potential = (last_close - last_vwma_fast) / last_close * 100
    return cur_trend, interest, potential


def get_state(security: ISSSecurity, tf: str, data_h1_vwma: pd.DataFrame) -> AssetState:
    dataset = data_h1_vwma.tail(256)
    has_data = not data_h1_vwma.empty
    trend = None
    interest = False
    potential = 0.0
    img = None
    if has_data:
        trend, interest, potential = get_potential(dataset)
        img = functions.create_image(dataset, True)
    return AssetState(security, tf, has_data, trend, img, interest, potential)


async def get_security_state(session: aiohttp.ClientSession, security: ISSSecurity) -> ScannerResult:
    # 2 month for 1H ( 6 to be converted to 4H)
    from_h1_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    # 1 year for 1D
    from_d1_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    data_d1 = await functions.get_stock_candles(session, security.get_ticker(), "D1", from_d1_date, None)
    data_h1 = await functions.get_stock_candles(session, security.get_ticker(), "H1", from_h1_date, None)
    data_h4 = pd.DataFrame()
    if not data_h1.empty:
        data_h4 = functions.aggregate(data_d1, 240)
    data_h1_vwma = pd.DataFrame()
    data_h4_vwma = pd.DataFrame()
    data_d1_vwma = pd.DataFrame()
    if not data_h1.empty:
        data_h1_vwma = functions.get_vwma(data_h1, 100, 200)
    if not data_h4.empty:
        data_h4_vwma = functions.get_vwma(data_h4, 100, 200)
    if not data_d1.empty:
        data_d1_vwma = functions.get_vwma(data_d1, 100, 200)
    return ScannerResult(security,
                         get_state(security, "H1", data_h1_vwma),
                         get_state(security, "H4", data_h4_vwma),
                         get_state(security, "D1", data_d1_vwma))


def print_scanner_results(results: List[ScannerResult]):
    scan_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_count = len(results)
    parent_dir = os.path.join("_scan", "out")  # os.path.join("_scan", scan_date)
    if not os.path.exists(parent_dir): os.makedirs(parent_dir)
    filtered_results = [x for x in results if x.get_h1_state().get_status() and x.get_h4_state().get_status() and x.get_d1_state().get_status()]
    total_with_data_count = len(filtered_results)
    interesting_results = [x for x in filtered_results if x.get_h1_state().get_interest() or x.get_h4_state().get_interest() or x.get_d1_state().get_interest()]
    interesting_results_count = len(interesting_results)
    # Sorting D1 potential -> H4 potential -> H1 potential -> list level -> ticker
    filtered_results = sorted(interesting_results,
                              key=lambda a: (
                                  max(a.get_d1_state().get_potential(),
                                      a.get_h4_state().get_potential(),
                                      a.get_h1_state().get_potential()),
                                  a.get_security().get_list_level(),
                                  a.get_security().get_ticker()
                              ),
                              reverse=True)

    res_params = []
    for res in filtered_results:
        res_params.append({
            'name': res.get_security().get_ticker(),
            'description': res.get_security().get_name(),
            'list_level': res.get_security().get_list_level(),
            'trend_h1': res.get_h1_state().get_trend(),
            'trend_h4': res.get_h4_state().get_trend(),
            'trend_d1': res.get_d1_state().get_trend(),
            'style_h1': res.get_h1_state().get_style(),
            'style_h4': res.get_h4_state().get_style(),
            'style_d1': res.get_d1_state().get_style(),
            'potential_h1': res.get_h1_state().get_potential_str(),
            'potential_h4': res.get_h4_state().get_potential_str(),
            'potential_d1': res.get_d1_state().get_potential_str(),
            'img_h1': res.get_h1_state().get_image_path(parent_dir),
            'img_h4': res.get_h4_state().get_image_path(parent_dir),
            'img_d1': res.get_d1_state().get_image_path(parent_dir)
        })

    with open('scanner_template.tmpl', 'r') as file:
        template = file.read()
    output = chevron.render(template=template, data={
        'total_scanned': results_count,
        'total_with_data': total_with_data_count,
        'interesting_results': interesting_results_count,
        'scan_list': res_params,
        'scan_date': scan_date
    })
    out_file_path = os.path.join(parent_dir, f"index.html")
    out_file = open(out_file_path, "w", encoding="utf-8")
    n = out_file.write(output)
    out_file.close()
    print(f"Found {interesting_results_count} interesting results\n")
    print(f"Result is written to {out_file_path}")


async def stock_screen():
    connector = aiohttp.TCPConnector(force_close=True, limit=50, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        securities = await get_moex_securities(session)
        print(f"Found {len(securities)} tickers on MOEX")
        tasks = []
        for security in securities:
            tasks.append(asyncio.create_task(get_security_state(session, security)))
        results = await asyncio.gather(*tasks)
        print_scanner_results(results)
        end = time.time()
        total_time = end - start
        print(f"Execution completed in {str(total_time)} sec")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()  # создаем цикл
    task = loop.create_task(  # в цикл добавляем 1 задачу
        stock_screen()
    )
    loop.run_until_complete(task)  # ждем окончания выполнения цикла
