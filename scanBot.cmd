if not DEFINED IS_MINIMIZED set IS_MINIMIZED=1 && start /min "" "%~dpnx0" %* && exit
cd C:\Work\finam\Finam-NN-Trade-Robot
python scanBot.py
exit