import multiprocessing
import time
import click
import os
from lib.train import train
from lib.search import self_play
from lib.process import MyPool
import signal


# @click.command()
# @click.option("--folder", default=-1)
# @click.option("--version", default=False)
def main(folder=-1, version=False):
    # Start method for PyTorch
    # multiprocessing.set_start_method('spawn')

    # Create folder name if not provided
    if folder == -1:
        current_time = str(int(time.time()))
    else:
        current_time = str(folder)

    # Catch SIGNINT
    # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    # signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        self_play_proc = pool.apply_async(self_play, args=(current_time, version,))
        train_proc = pool.apply_async(train, args=(current_time, version,))
        # train(current_time, version)
        # Comment one line or the other to get the stack trace
        # Must add a loooooong timer otherwise signals are not caught
        self_play_proc.get(25200)
        # train_proc.get(18000)

    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main("1586332421", 38)


