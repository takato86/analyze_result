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
    n_episodes = 1000
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


def load_first_steps(file_pattern):
    first_steps = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            row = next(reader)
            first_steps.append(int(row[0]))
    return first_steps


def load_asymptotic_performances(file_pattern):
    n_window = 10
    n_episodes = 1000
    asymptotic_performances = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            asymptotic_performance = 0
            for step, row in enumerate(reader):
                if step >= n_episodes - n_window:
                    asymptotic_performance += int(row[0])
            asymptotic_performances.append(asymptotic_performance / n_window)
    return asymptotic_performances


def load_asymptotic_steps(file_pattern):
    asymptotic_steps = []
    n_window = 10
    n_episodes = 1000
    for run, file_path in enumerate(glob.glob(file_pattern)):
        data = []
        asymptotic_performance = 0
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for step, row in enumerate(reader):
                data.append(row)
                if step >= n_episodes - n_window:
                    asymptotic_performance += int(row[0])
            asymptotic_performance /= n_window
            for step, row in enumerate(data):
                if asymptotic_performance >= int(row[0]):
                    asymptotic_steps.append(step)
                    break
    return asymptotic_steps


def export_csv(file_name, contents):
    if type(contents) != pd.DataFrame:
        export_df = pd.DataFrame(contents, columns=sys.argv[1:])
    else:
        export_df = contents
        export_df.columns = sys.argv[1:]
    export_df.to_csv(file_name, index=False)
    logger.info(f"export {file_name}")


def calc_cumulative_steps(file_pattern, threshold):
    cumulative_steps = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            cumulative_step = 0
            for step, row in enumerate(reader):
                if int(row[0]) <= threshold:
                    cumulative_step += int(row[0])
            cumulative_steps.append(cumulative_step)
    return cumulative_steps


def calc_time_to_threshold(file_pattern, threshold):
    time_to_thresholds = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            time_to_threshold = 1000
            for step, row in enumerate(reader):
                if int(row[0]) < threshold:
                    time_to_threshold = step + 1
                    break
            time_to_thresholds.append(time_to_threshold)
    return time_to_thresholds


def transform(contents):
    return pd.DataFrame(contents).T


def main():
    argvs = sys.argv[1:]
    mean_y = []
    first_steps = []
    asymptotic_performance = []
    asymptotic_steps = []
    cumulative_steps = []
    time_to_threshold_700 = []
    time_to_threshold_500 = []
    time_to_threshold_300 = []
    time_to_threshold_100 = []
    time_to_threshold_50 = []
    for argv in argvs:
        file_pattern = argv+"*"
        mean_y.append(load_mean_steps(file_pattern))
        first_steps.append(load_first_steps(file_pattern))
        asymptotic_performance.append(load_asymptotic_performances(file_pattern))
        asymptotic_steps.append(load_asymptotic_steps(file_pattern))
        cumulative_steps.append(calc_cumulative_steps(file_pattern, 50))
        time_to_threshold_700.append(calc_time_to_threshold(file_pattern, 700))
        time_to_threshold_500.append(calc_time_to_threshold(file_pattern, 500))
        time_to_threshold_300.append(calc_time_to_threshold(file_pattern, 300))
        time_to_threshold_100.append(calc_time_to_threshold(file_pattern, 100))
        time_to_threshold_50.append(calc_time_to_threshold(file_pattern, 50))
    total_steps = np.array([get_total_steps(mean_y)])
    mean_y = np.array(mean_y).T.tolist()
    first_steps = pd.DataFrame(first_steps).T
    asymptotic_performance = transform(asymptotic_performance)
    asymptotic_steps = transform(asymptotic_steps)
    cumulative_steps = transform(cumulative_steps)
    time_to_threshold_700 = transform(time_to_threshold_700)
    time_to_threshold_500 = transform(time_to_threshold_500)
    time_to_threshold_300 = transform(time_to_threshold_300)
    time_to_threshold_100 = transform(time_to_threshold_100)
    time_to_threshold_50 = transform(time_to_threshold_50)
    export_csv("total_steps.csv", total_steps)
    export_csv("mean_steps.csv", mean_y)
    export_csv("first_steps.csv", first_steps)
    export_csv("asymptotic_performance.csv", asymptotic_performance)
    export_csv("asymptotic_steps.csv", asymptotic_steps)
    export_csv("cumulative_steps.csv", cumulative_steps)
    export_csv("time_to_threshold_700.csv", time_to_threshold_700)
    export_csv("time_to_threshold_500.csv", time_to_threshold_500)
    export_csv("time_to_threshold_300.csv", time_to_threshold_300)
    export_csv("time_to_threshold_100.csv", time_to_threshold_100)
    export_csv("time_to_threshold_50.csv", time_to_threshold_50)
if __name__ == "__main__":
    main()