import csv
import os
import json
import pandas as pd
import numpy as np
import sys
import glob
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def load_first_steps(file_pattern):
    first_steps = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            row = next(reader)
            first_steps.append(int(row[0]))
    return first_steps


def load_asymptotic_performances(file_pattern, n_window=10, n_episodes=1000):
    asymptotic_performances = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            asymptotic_performance = 0
            episode_index = n_episodes - 1
            for step, row in enumerate(reader):
                if step > episode_index - n_window and step <= episode_index:
                    asymptotic_performance += int(row[0])
            asymptotic_performances.append(asymptotic_performance / n_window)
    return asymptotic_performances


def load_asymptotic_steps(file_pattern, n_window=10, n_episodes=1000):
    asymptotic_steps = []
    for run, file_path in enumerate(glob.glob(file_pattern)):
        data = []
        asymptotic_performance = 0
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for step, row in enumerate(reader):
                episode_index = n_episodes - 1
                data.append(row)
                if step > episode_index - n_window and step <= episode_index:
                    asymptotic_performance += int(row[0])
            asymptotic_performance /= n_window
            for step, row in enumerate(data):
                if asymptotic_performance >= int(row[0]):
                    asymptotic_steps.append(step)
                    break
    return asymptotic_steps


def export_csv(file_name, contents):
    dir_name = "out"
    if type(contents) != pd.DataFrame:
        export_df = pd.DataFrame(contents, columns=sys.argv[1:])
    else:
        export_df = contents
        export_df.columns = sys.argv[1:]
    export_df.to_csv(os.path.join(dir_name, file_name), index=False)
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


def result_learning_curve(file_pattern, prefix):
    logger.info("Loading {}...".format(file_pattern))
    serieses = [
        pd.read_csv(file_path, encoding="utf-8")
        for file_path in glob.glob(file_pattern)
    ]
    logger.info("Complete loading.")
    df = pd.concat(serieses, axis=1)
    learning_curve = pd.DataFrame()
    learning_curve["mean"] = df.mean(axis=1)
    learning_curve["se"] = (df.var(axis=1) / len(df)) ** 0.5
    learning_curve["upper"] = learning_curve["mean"] + learning_curve["se"]
    learning_curve["lower"] = learning_curve["mean"] - learning_curve["se"]
    learning_curve = learning_curve.add_prefix(prefix)
    return learning_curve


def transform(contents):
    return pd.DataFrame(contents).T


def main():
    with open("config.json", "r") as f:
        configs = json.load(f)

    file_patterns = configs["file_patterns"]
    prefixes = configs["prefixes"]
    # first_steps = []
    # asymptotic_performance = []
    # asymptotic_steps = []
    # cumulative_steps = []
    # time_to_threshold_700 = []
    # time_to_threshold_500 = []
    # time_to_threshold_300 = []
    # time_to_threshold_100 = []
    # time_to_threshold_50 = []
    learning_curves = []
    for file_pattern, prefix in zip(file_patterns, prefixes):
        learning_curves.append(
            result_learning_curve(file_pattern, prefix)
        )
        # first_steps.append(load_first_steps(file_pattern))
        # asymptotic_performance.append(load_asymptotic_performances(file_pattern))
        # asymptotic_steps.append(load_asymptotic_steps(file_pattern))
        # cumulative_steps.append(calc_cumulative_steps(file_pattern, 50))
        # time_to_threshold_700.append(calc_time_to_threshold(file_pattern, 700))
        # time_to_threshold_500.append(calc_time_to_threshold(file_pattern, 500))
        # time_to_threshold_300.append(calc_time_to_threshold(file_pattern, 300))
        # time_to_threshold_100.append(calc_time_to_threshold(file_pattern, 100))
        # time_to_threshold_50.append(calc_time_to_threshold(file_pattern, 50))
    learning_curve_df = pd.concat(learning_curves, axis=1)
    learning_curve_df.to_csv(os.path.join("out",
                                          "fourrooms_learning_curves.csv"))
    logging.info("Output {}.".format(
        os.path.join("out", "fourrooms_learning_curves.csv"))
    )
    # first_steps = pd.DataFrame(first_steps).T
    # asymptotic_performance = transform(asymptotic_performance)
    # asymptotic_steps = transform(asymptotic_steps)
    # cumulative_steps = transform(cumulative_steps)
    # time_to_threshold_700 = transform(time_to_threshold_700)
    # time_to_threshold_500 = transform(time_to_threshold_500)
    # time_to_threshold_300 = transform(time_to_threshold_300)
    # time_to_threshold_100 = transform(time_to_threshold_100)
    # time_to_threshold_50 = transform(time_to_threshold_50)
    # export_csv("first_steps.csv", first_steps)
    # export_csv("asymptotic_performance.csv", asymptotic_performance)
    # export_csv("asymptotic_steps.csv", asymptotic_steps)
    # export_csv("cumulative_steps.csv", cumulative_steps)
    # export_csv("time_to_threshold_700.csv", time_to_threshold_700)
    # export_csv("time_to_threshold_500.csv", time_to_threshold_500)
    # export_csv("time_to_threshold_300.csv", time_to_threshold_300)
    # export_csv("time_to_threshold_100.csv", time_to_threshold_100)
    # export_csv("time_to_threshold_50.csv", time_to_threshold_50)


if __name__ == "__main__":
    main()
