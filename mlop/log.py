import builtins
import logging
import sys
import time

from .iface import make_compat_message_v1
from .util import ANSI

builtins_input = builtins.input


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
    def __init__(
        self, logger, queue, level=logging.INFO, stream=sys.stdout, type="stdout"
    ):
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
            self.queue.put(
                make_compat_message_v1(self.level, m, int(time.time()), self.count)
            )
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
