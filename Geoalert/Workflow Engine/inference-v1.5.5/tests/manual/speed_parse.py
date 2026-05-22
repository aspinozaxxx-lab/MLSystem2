from time import time
from urban.base.compose import parse_config
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

def speed_parse():
    path = 'tests/test_pipelines/sample.txt'
    times = list()
    for run in range(500):
        t0 = time()

        for _ in range(100):
            config = parse_config(path)

        t1 = time()
        times.append(t1-t0)
    plt.plot(times)
    plt.show()

speed_parse()