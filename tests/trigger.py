import math
import random
import time

from .args import init_test, timer

TAG = "trigger"


@timer
def test_trigger(mlop, run):
    NUM_EPOCHS = 50_000
    ITEM_PER_EPOCH = 20
    WAIT_INT = 1_000
    WAIT = ITEM_PER_EPOCH * 0.05

    decay_rate = 0.0001
    noise_scale = 0.5

    for i in range(NUM_EPOCHS):
        base_value = math.exp(-decay_rate * i)
        run.log({f"val/{TAG}": base_value + (random.random() - 0.5) * noise_scale})
        run.log({f"{TAG}-total": i + 1})
        if i % WAIT_INT == 0:
            print(f"{TAG}: Epoch {i + 1} / {NUM_EPOCHS}, sleeping {WAIT}s")
            time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_trigger(mlop, run)
