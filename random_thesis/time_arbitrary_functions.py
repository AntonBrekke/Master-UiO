import concurrent.futures
import time
from tqdm import tqdm

def timed_future_progress_bar(future, expected_time, increments=10):
    """
    Display progress bar for expected_time seconds.
    Complete early if future completes.
    Wait for future if it doesn't complete in expected_time.
    """
    interval = expected_time / increments
    with tqdm(total=increments) as pbar:
        for i in range(increments - 1):
            if future.done():
                # finish the progress bar
                # not sure if there's a cleaner way to do this?
                pbar.update(increments - i)
                return
            else:
                time.sleep(interval)
                pbar.update()
        # if the future still hasn't completed, wait for it.
        future.result()
        pbar.update()

def blocking_job():
    time.sleep(5)
    return 'result'


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(blocking_job)
        timed_future_progress_bar(future, 5, 100)
    print(f'Work done: {future.result()}')

main()
