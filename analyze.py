import csv
import os
import pandas as pd
import numpy as np
import sys
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_mean_steps(file_pattern):
    n_episodes = 2000
    mean_y = [0] * n_episodes
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for step, row in enumerate(reader):
                mean_y[step] = 1/(1+run) * (run * mean_y[step] + int(row[0]))
    return mean_y

def get_total_steps(mean_ys):
    total_steps = []
    for mean_y in mean_ys:
        total_steps.append(sum(mean_y))
    return total_steps

def export_csv(file_name, contents):
    export_df = pd.DataFrame(contents, columns=sys.argv[1:])
    export_df.to_csv(file_name, index=False)
    logger.info(f"export {file_name}")

def main():
    argvs = sys.argv[1:]
    mean_y = []
    for argv in argvs:
        file_pattern = argv+"*"
        mean_y.append(load_mean_steps(file_pattern))
    total_steps = np.array([get_total_steps(mean_y)])
    mean_y = np.array(mean_y).T.tolist()
    export_csv("total_steps.csv", total_steps)
    export_csv("mean_steps.csv", mean_y)
    


if __name__ == "__main__":
    main()