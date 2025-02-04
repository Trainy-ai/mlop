import logging
import platform

import psutil

from .util import to_human

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "System"


class System:
    def __init__(self):
        self.uname = platform.uname()._asdict()

        self.cpu_count = psutil.cpu_count
        self.cpu_freq = [i._asdict() for i in psutil.cpu_freq(percpu=True)]
        self.cpu_freq_min = min([i["min"] for i in self.cpu_freq])
        self.cpu_freq_max = max([i["max"] for i in self.cpu_freq])
        self.svmem = psutil.virtual_memory()._asdict()
        self.sswap = psutil.swap_memory()._asdict()
        self.disk = [i._asdict() for i in psutil.disk_partitions()]
        self.net_if_addrs = {
            i: [
                {k: v for k, v in j._asdict().items() if k != "family"}
                for j in psutil.net_if_addrs()[i]
            ]
            for i in psutil.net_if_addrs()
        }

        self.boot_time = psutil.boot_time()
        self.users = [i._asdict() for i in psutil.users()]

        self.gpu = self.get_gpu()

    def __getattr__(self, name):
        return self.get_psutil(name)

    def get_psutil(self, name):  # handling os specific methods
        if hasattr(psutil, name):
            return getattr(psutil, name)
        else:
            return None

    def get_gpu(self):
        d = {}
        try:
            import pynvml

            try:
                pynvml.nvmlInit()
                logger.info(f"{tag}: NVIDIA GPU detected")
                d["nvidia"] = {
                    "count": pynvml.nvmlDeviceGetCount(),
                    "driver": pynvml.nvmlSystemGetDriverVersion(),
                }
                for i in range(d["nvidia"]["count"]):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    d["nvidia"][i] = {
                        "name": pynvml.nvmlDeviceGetName(handle),
                        "memory": {
                            "total": to_human(
                                pynvml.nvmlDeviceGetMemoryInfo(handle).total
                            ),
                        },
                        "temp": pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                    }
            except pynvml.NVMLError_LibraryNotFound:
                logger.debug(f"{tag}: NVIDIA driver not found")
            except Exception as e:
                logger.error("%s: NVIDIA error: %s", tag, e)
        except ImportError:
            logger.debug(f"{tag}: pynvml not found")
        return d

    def info(self, debug=False):
        d = {
            "platform": self.uname,
            "cpu": {
                "physical": self.cpu_count(logical=False),
                "virtual": self.cpu_count(logical=True),
                "freq": {
                    "min": self.cpu_freq_min,
                    "max": self.cpu_freq_max,
                },
            },
            "memory": {
                "virtual": to_human(self.svmem["total"]),
                "swap": to_human(self.sswap["total"]),
            },
            "boot_time": self.boot_time,
        }
        if self.gpu:
            d["gpu"] = self.gpu
        if debug:
            d = {
                **d,
                "disk": self.disk,
                "network": self.net_if_addrs,
                "users": self.users,
            }
        return d
