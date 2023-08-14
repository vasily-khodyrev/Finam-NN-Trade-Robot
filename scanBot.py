import json
import os
import signal
import subprocess
import time

import telebot

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
    send_last_results("Stocks", message)


@bot.message_handler(commands=['last_futures'])
def send_last_futures_result(message: telebot.types.Message):
    send_last_results("Futures", message)


def send_last_results(_type: str,
                      message: telebot.types.Message):
    parent_dir = os.path.join("_scan", "out_" + _type)
    out_json_path = os.path.join(parent_dir, f"data.json")
    if not os.path.exists(out_json_path):
        bot.reply_to(message, "Ups... Something went wrong")
        return
    data_file = open(out_json_path)
    data = json.load(data_file)
    data_file.close()
    total_scanned = data["total_scanned"]
    interesting_results = data["interesting_results"]
    scan_date = data["scan_date"]
    scan_list = data["scan_list"]
    general_msg = f"Total scanned {total_scanned}. Interesting: {interesting_results} ScanDate: {scan_date}"
    bot.send_message(message.from_user.id, general_msg)
    if int(interesting_results) > 0:
        for i in range(0, 3):
            images = []
            if i < len(scan_list):
                stock = scan_list[i]
                s_name = stock["name"]
                s_desc = stock["description"]
                s_price = stock["last_price"]

                _str = f"{s_name}:{s_desc} p:{s_price} "
                states = stock["states"]
                for state in states:
                    s_img = state["img"]
                    s_tf = state["tf"]
                    s_potential = state["potential"]
                    _str = f"{_str} {s_tf}:{s_potential}"
                    _img_str = f"{s_name}: price:{s_price} {s_tf}:{s_potential}"
                    if s_img:
                        images.append(
                            telebot.types.InputMediaPhoto(open(s_img, 'rb'), caption=_img_str)
                        )
                bot.send_message(message.from_user.id, _str)
                if len(images) > 0:
                    bot.send_media_group(message.from_user.id, images)


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Terminate script"
    print(msg, end="", flush=True)
    exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    bot.infinity_polling()
