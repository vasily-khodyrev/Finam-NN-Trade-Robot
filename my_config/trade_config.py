# конфигурационный файл для торговой стратегии

class Config:

    root_folder = "NN_futures"
    futures = "SI"
    training_NN = {"SiH1", "SiM1", "SiU1", "SiZ1", "SiH2", "SiM2", "SiU2", "SiZ2", "SiH3", "SiM3", "SiU3"}  # тикеры по которым обучаем нейросеть
    portfolio = ["SiH1", "SiM1", "SiU1", "SiZ1", "SiH2", "SiM2", "SiU2", "SiZ2", "SiH3", "SiM3", "SiU3"]  # тикеры по которым торгуем и скачиваем исторические данные
    security_board = "RFUD"  # класс тикеров

    # доступные M1, M10, H1
    data_time_frames = ["M1", "M5", "M15", "M30"]  # timeframes to load data
    timeframe_0 = "M1"  # таймфрейм для обучения нейросети - вход и на этом же таймфрейме будем торговать
    timeframe_1 = "M10"  # таймфрейм для обучения нейросети - выход
    start = "2021-01-01"  # с какой даты загружаем исторические данные с MOEX

    trading_hours_start = "10:00"  # время работы биржи - начало
    trading_hours_end = "23:50"  # время работы биржи - конец

    # параметры для отрисовки картинок
    period_vwma_slow = 200  # период медленной SMA
    period_vwma_fast = 100  # период быстрой SMA
    period_vwma_vfast = 50  # период супер быстрой SMA
    look_forward = 60  # на сколько свечек вперед оценивать результат
    draw_window = 256  # окно данных
    steps_skip = 16  # шаг сдвига окна данных
    draw_size = 256  # размер стороны квадратной картинки
