import numpy as np
import pandas as pd
import argparse
from utils import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data", help="data file")
    args = parser.parse_args()
    data = args.data

    doa = compute_doa(data)
    print("DOA:", doa)