import builtins
import logging
import os
import sys
import time

from .iface import make_compat_message_v1
from .sets import get_console

builtins_input = builtins.input


class ANSI:
    base = "\033["  # "\x1b["
    reset = f"{base}0m"
    bold = f"{base}1m"
    faint = f"{base}2m"
    italic = f"{base}3m"
    underline = f"{base}4m"
    slow_blink = f"{base}5m"
    rapid_blink = f"{base}6m"

    black = f"{base}30m"
    red = f"{base}31m"
    green = f"{base}32m"
    yellow = f"{base}33m"
    blue = f"{base}34m"
    purple = f"{base}35m"
    cyan = f"{base}36m"
    white = f"{base}37m"

    if not __import__("sys").stdout.isatty() and get_console() == "python":
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        if __import__("platform").system() == "Windows":
            os.system("")


class ColorFormatter(logging.Formatter):
    def format(self, record):
        prefix = ANSI.bold + ANSI.blue + f"{__name__.split('.')[0]}:" + ANSI.reset
        colors = {
            "DEBUG": ANSI.green,
            "INFO": ANSI.blue,
            "WARNING": ANSI.yellow,
            "ERROR": ANSI.red,
            "CRITICAL": ANSI.purple,
        }
        styles = {
            "DEBUG": " 💬 ",
            "INFO": " 🚀 ",
            "WARNING": " 🚨 ",
            "ERROR": " ⛔ ",
            "CRITICAL": " 🚫 ",
        }
        color = colors.get(record.levelname, "")
        style = styles.get(record.levelname, "")
        # record.msg = f"{color}{record.msg}{ANSI.RESET}"
        return f"{prefix}{color}{style}{super().format(record)}{ANSI.reset}"


class ConsoleHandler:
    def __init__(self, logger, queue, level=logging.INFO, stream=sys.stdout, type="stdout"):
        self.logger = logger
        self.queue = queue
        self.level = level
        self.stream = stream
        self.type = type
        self.count = 0

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.count += 1
            m = line.rstrip()
            self.queue.put(make_compat_message_v1(self.level, m, int(time.time()), self.count))
            self.logger.log(self.level, m)
        self.stream.write(buf)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def input_hook(prompt="", logger=None):
    content = builtins_input(prompt)
    logger.warn(f"{prompt}{content}")
    return content
