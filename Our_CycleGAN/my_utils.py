from contextlib import contextmanager
import time

@contextmanager
def timer(code_part = "No part name"):
    start = time.perf_counter()
    yield  # Everything in the `with` block executes here
    end = time.perf_counter()
    take_time = (end - start)*1000
    print(f"*** {code_part} : ({take_time:.2f} ms) ({take_time/1000:.2f} s) ({take_time/60000:.2f} min) ({take_time/3600000:.2f} hr) ***")