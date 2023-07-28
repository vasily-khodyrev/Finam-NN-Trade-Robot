"""
    В этом коде мы асинхронно получаем исторические данные с MOEX
    и сохраняем их в CSV файлы. Т.к. получаем их бесплатно, то
    есть задержка в полученных данных на 15 минут.
    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""


import asyncio
import os
from datetime import datetime

import aiohttp
import aiomoex

import functions
from my_config.trade_config import Config  # Файл конфигурации торгового робота


async def get_futures_candles(root_folder, session, ticker, timeframes, start, end):
    for timeframe in timeframes:
        df = functions.get_futures_candles(session, ticker, timeframe, start, end)
        df.to_csv(os.path.join(root_folder, "csv", f"{ticker}_{timeframe}.csv"), index=False, encoding='utf-8', sep=',')
        print(f"{ticker} {timeframe}: received")
        #print(df)


async def get_all_historical_candles(portfolio, timeframes, start, end):
    """Запуск асинхронной задачи получения исторических данных для каждого тикера из портфеля."""
    async with aiohttp.ClientSession() as session:
        strategy_tasks = []
        for instrument in portfolio:
            strategy_tasks.append(asyncio.create_task(get_futures_candles(session, instrument, timeframes, start, end)))
        await asyncio.gather(*strategy_tasks)


async def get_security(session, ticker):
    url = aiomoex.request_helpers.make_url(security=ticker)
    query = aiomoex.request_helpers.make_query()
    data = await aiomoex.request_helpers.get_short_data(session, url, "description", query)
    first_date = ""
    last_date = ""
    for line in data:
        if line['name'] == "FRSTTRADE":
            first_date = line['value']
        if line['name'] == "LSTTRADE":
            last_date = line['value']
    print(f"{ticker}: {first_date} - {last_date}")
    return {"start": first_date, "end": last_date}


async def get_securities(root_folder, tickers, timeframes, start, end):
    """Функция получения инструмента с MOEX."""
    async with aiohttp.ClientSession() as session:
        start_date_string = start
        end_date = datetime.strptime(end, "%Y-%m-%d")
        for ticker in tickers:
            valid_time = await get_security(session, ticker)
            ticker_start = valid_time["start"]
            ticker_end = valid_time["end"]
            ticker_start_date = datetime.strptime(ticker_start, "%Y-%m-%d")
            ticker_end_date = datetime.strptime(ticker_end, "%Y-%m-%d")
            start_date = datetime.strptime(start_date_string, "%Y-%m-%d")
            if start_date > ticker_start_date:
                ticker_start = start_date_string
            if end_date < ticker_end_date:
                ticker_end = end
            if start_date > datetime.now():
                break
            print(f"Getting candles for {ticker}: {ticker_start} - {ticker_end}")
            await get_futures_candles(root_folder, session, ticker, timeframes, ticker_start, ticker_end)
            start_date_string = ticker_end


if __name__ == "__main__":

    # применение настроек из config.py
    root_folder = Config.root_folder #основная папка для выходных данных
    portfolio = Config.portfolio  # тикеры по которым скачиваем исторические данные
    futures = Config.futures # Главный фьючесный символ по которому мы будем собирать <>! Continuous futures данные
    timeframe_0 = Config.timeframe_0  # таймфрейм для обучения нейросети - вход
    timeframe_1 = Config.timeframe_1  # таймфрейм для обучения нейросети - выход
    start = Config.start  # с какой даты загружаем исторические данные с MOEX
    end = datetime.now().strftime("%Y-%m-%d")  # по сегодня
    
    # создаем необходимые каталоги
    functions.create_some_folders(timeframes=[timeframe_0, timeframe_1], root_folder=root_folder)

    # запуск асинхронного цикла получения исторических данных
    loop = asyncio.get_event_loop()  # создаем цикл
    task = loop.create_task(  # в цикл добавляем 1 задачу
        get_securities(
            root_folder=root_folder,
            tickers=portfolio,
            timeframes=[timeframe_0, timeframe_1],
            start=start,
            end=end
        )
        #get_all_historical_candles(  # запуск получения исторических данных с MOEX
        #    portfolio=portfolio,
        #    timeframes=[timeframe_0, timeframe_1],  # по каким таймфреймам скачиваем данные
        #    start=start,
        #    end=end,
        #)
    )
    loop.run_until_complete(task)  # ждем окончания выполнения цикла 
