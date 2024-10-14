import ray
import time
import torch
from ray.util.multiprocessing import Pool
#from yolov10_model import YOLOv10

ray.init()

@ray.remote
def f(i):
    time.sleep(1)
    return i

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))
pool = Pool(ray_address="177.205.85.62:")