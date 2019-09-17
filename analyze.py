import csv
import os
import pandas as pd
import numpy as np
import sys
import glob


def load_mean_steps(file_pattern):
    n_episodes = 2000
    mean_y = [0] * n_episodes
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for step, row in enumerate(reader):
                mean_y[step] = 1/(1+run) * (run * mean_y[step] + int(row[0]))
    return mean_y

def main():
    argvs = sys.argv[1:]
    mean_y = []
    for argv in argvs:
        file_pattern = argv+"*"
        mean_y.append(load_mean_steps(file_pattern))
    import pdb; pdb.set_trace()
    mean_y = np.array(mean_y).T.tolist()
    export_df = pd.DataFrame(mean_y, columns=argvs)
    export_df.to_csv("mean_steps.csv", index=False)


if __name__ == "__main__":
    main()