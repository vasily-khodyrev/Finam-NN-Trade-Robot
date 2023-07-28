"""
    В этом коде мы подготавливаем данные для обучения нейросети.
    Генерируем картинки по следующему алгоритму:
    1. берем картинку графика цены закрытия + SMA1 + SMA2 за определенный
    интервал на таймфрейме_0
    2. если закрытие на старшем таймфрейме_1 > закрытия предыдущей свечи старшего
    таймфрейма_1, то для этой картинки назначаем класс 1, иначе класс 0
    P.S. SMA1, SMA2 - скользящие средние
    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""


import functions
import functions_nn
import os
import matplotlib.pyplot as plt
from my_config.trade_config import Config  # Файл конфигурации торгового робота


def is_next_profit(cur_date, date_to_index, close_data, high_data, low_data):
    profit_level = 0.005  # 0.5% profit
    max_loss = 0.001      # 0.1% loss
    check_range = 20
    date_index = date_to_index[cur_date]
    data_length = len(close_data)
    result = False
    current_close = close_data[date_index]
    profit_price = current_close * (1 + profit_level)
    max_loss_price = current_close * (1 - max_loss)
    for idx in range(date_index, date_index+check_range):
        if idx > data_length - 1:
            break
        next_low = low_data[idx]
        next_high = high_data[idx]
        if next_low < max_loss_price: #
            result = False
            break
        if next_high > profit_price:
            result = True
            break
    return result


if __name__ == "__main__":
    # применение настроек из config.py
    root_folder = Config.root_folder  # основная папка для выходных данных
    portfolio = Config.training_NN  # тикеры по которым обучаем нейросеть
    timeframe_0 = Config.timeframe_0  # таймфрейм для обучения нейросети - вход
    timeframe_1 = Config.timeframe_1  # таймфрейм для обучения нейросети - выход

    # параметры для отрисовки картинок
    period_vwma_slow = Config.period_vwma_slow  # период медленной SMA
    period_vwma_fast = Config.period_vwma_fast  # период быстрой SMA
    draw_window = Config.draw_window  # окно данных
    steps_skip = Config.steps_skip  # шаг сдвига окна данных
    draw_size = Config.draw_size  # размер стороны квадратной картинки

    # создаем необходимые каталоги
    functions.create_some_folders(timeframes=[timeframe_0], classes=["0", "1"], root_folder=root_folder)

    for ticker in portfolio:

        # считываем данные для обучения нейросети - выход - timeframe_1
        df_out = functions_nn.get_df_t1(root_folder, ticker, timeframe_1)
        # print(df_out)
        _date_out = df_out["datetime"].tolist()
        _date_out_index = {_date_out[i]: i for i in range(len(_date_out))}  # {дата : индекс}
        _close_out = df_out["close"].tolist()
        _high_out = df_out["high"].tolist()
        _low_out = df_out["low"].tolist()

        # считываем данные для обучения нейросети - вход - timeframe_0
        df_in = functions_nn.get_df_tf0(root_folder, ticker, timeframe_0, period_vwma_fast, period_vwma_slow)
        # print(df_in)
        _date_in = df_in["datetime"].tolist()
        _close_in = df_in["close"].tolist()
        vwma_fast = df_in["vwma_fast"].tolist()
        vwma_slow = df_in["vwma_slow"].tolist()

        # # вывод на график Close + SMA последних 200 значений
        # df_in[['close', 'sma_fast', 'sma_slow']].iloc[-200:].plot(label='df', figsize=(16, 8))
        # plt.show()

        _steps, j = 0, 0
        # рисуем картинки только для младшего ТФ
        for _date in _date_in:
            if _date in _date_out:  # если дата младшего ТФ есть в датах старшего ТФ
                _steps += 1
                j += 1
                if _steps >= steps_skip and j >= draw_window:
                    _steps = 0

                    # формируем картинку для нейросети с привязкой к дате и тикеру с шагом steps_skip
                    # размером [draw_size, draw_size]
                    _sma_fast_list = vwma_fast[j - draw_window:j]
                    _sma_slow_list = vwma_slow[j - draw_window:j]
                    _closes_list = _close_in[j-draw_window:j]

                    # генерация картинки для обучения/теста нейросети
                    img = functions_nn.generate_img(_sma_fast_list, _sma_slow_list, _closes_list, draw_window)
                    # img.show()  # показать сгенерированную картинку

                    _date_str = _date.strftime("%Y_%m_%d_%H_%M_%S")
                    _filename = f"{ticker}-{timeframe_0}-{_date_str}.png"
                    _path = os.path.join(root_folder, f"training_dataset_{timeframe_0}")

                    # проводим классификацию изображений (пока будем определять только рост)
                    # if data.close[0] > data.close[-1]:
                    if is_next_profit(_date, _date_out_index, _close_out, _high_out, _low_out):
                        _path = os.path.join(_path, "1")
                    else:
                        _path = os.path.join(_path, "0")

                    img.save(os.path.join(_path, _filename))
                print(ticker, _date)
