import json
import os
import time
from typing import Optional, Tuple
import signal

import telebot
import subprocess

BOT_TOKEN = os.environ.get('BOT_TOKEN')
OWNER_TELE_ID = int(os.environ.get('OWNER_TELE_ID'))

script_is_running = False

if BOT_TOKEN is None or len(BOT_TOKEN) == 0:
    print(f"Token not defined")
    exit(-1)

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['scan_stock'])
def scan_and_send(message: telebot.types.Message):
    start = time.time()
    print(f"scan_stock from user {message.from_user.id}")
    if message.from_user.id == OWNER_TELE_ID:
        global script_is_running
        if not script_is_running:
            script_is_running = True
            print(f"starting scanner script")
            command = ['cmd', '/c', 'python', 'scanner.py']
            process = subprocess.Popen(command)
            process.wait()
            end = time.time()
            total_time = end - start
            print(f"Script completed with return code {process.returncode} in {str(total_time)} sec")
            script_is_running = False
        else:
            print(f"Script is already running - skip.")
    else:
        bot.reply_to(message, "Sorry - you're not owner of this bot - you can't initiate scan")
    send_last_stock_result(message)


@bot.message_handler(commands=['scan_futures'])
def scan_and_send(message: telebot.types.Message):
    start = time.time()
    print(f"scan_futures from user {message.from_user.id}")
    if message.from_user.id == OWNER_TELE_ID:
        global script_is_running
        if not script_is_running:
            script_is_running = True
            print(f"starting scanner script")
            command = ['cmd', '/c', 'python', 'scanner.py', 'futures']
            process = subprocess.Popen(command)
            process.wait()
            end = time.time()
            total_time = end - start
            print(f"Script completed with return code {process.returncode} in {str(total_time)} sec")
            script_is_running = False
        else:
            print(f"Script is already running - skip.")
    else:
        bot.reply_to(message, "Sorry - you're not owner of this bot - you can't initiate scan")
    send_last_futures_result(message)


@bot.message_handler(commands=['last_stock'])
def send_last_stock_result(message: telebot.types.Message):
    res = get_images("Stocks")
    if res is None:
        bot.reply_to(message, "Ups... Something went wrong")
        return
    general_msg, images = res
    bot.reply_to(message, general_msg)
    for img in images:
        _msg, _img_path = img
        bot.send_photo(message.from_user.id, photo=open(_img_path, 'rb'), caption=_msg)


@bot.message_handler(commands=['last_futures'])
def send_last_futures_result(message: telebot.types.Message):
    res = get_images("Futures")
    if res is None:
        bot.reply_to(message, "Ups... Something went wrong")
        return
    general_msg, images = res
    bot.reply_to(message, general_msg)
    for img in images:
        _msg, _img_path = img
        bot.send_photo(message.from_user.id, photo=open(_img_path, 'rb'), caption=_msg)


def get_images(_type: str) -> Optional[Tuple[str, list[Tuple[str, str]]]]:
    parent_dir = os.path.join("_scan", "out_" + _type)
    out_json_path = os.path.join(parent_dir, f"data.json")
    if not os.path.exists(out_json_path):
        return None
    data_file = open(out_json_path)
    data = json.load(data_file)
    data_file.close()
    total_scanned = data["total_scanned"]
    interesting_results = data["interesting_results"]
    scan_date = data["scan_date"]
    scan_list = data["scan_list"]
    general_msg = f"Total scanned {total_scanned}. Interesting: {interesting_results} ScanDate: {scan_date}"
    tf_1 = data["tf_1"]
    images = []
    for i in range(0, 3):
        if i < len(scan_list):
            first_stock = scan_list[i]
            s_name = first_stock["name"]
            s_desc = first_stock["description"]
            s_price = first_stock["last_price"]
            s_img = first_stock["img_h1"]
            s_potential = first_stock["potential_h1"]
            images.append((f"{s_name}-{s_desc} Price: {s_price} Potential-{tf_1}: {s_potential}", s_img))
    return general_msg, images


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Terminate script"
    print(msg, end="", flush=True)
    exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    bot.infinity_polling()
