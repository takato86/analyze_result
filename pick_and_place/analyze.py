from typing import List
import pandas as pd
import numpy as np
import glob
import sys
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def read_files(file_pattern: str) -> List[pd.DataFrame]:
    dfs = []

    for fname in glob.glob(file_pattern):
        dfs.append(pd.read_csv(fname, index_col=0))

    return dfs


def export(out_file_path, content):
    l = max(map(lambda x: len(x[1]), content.items()))
    filled_content = {}
    for key, val in content.items():
        filled_content[key] = list(val) + [None]*(l-len(val))
    export_df = pd.DataFrame(filled_content)
    export_df.to_csv(out_file_path)
    logger.info(f"Export {out_file_path}")


def result_learning_curves(file_pattern, prefix, column="test/success_rate", window=10):
    """学習曲線を描画するための結果を返す。"""
    serieses = []
    result = pd.DataFrame()

    for file_path in glob.glob(file_pattern):
        logger.info("Loading {}".format(file_path))
        series = pd.read_csv(file_path, index_col=0)[column]
        series = series.dropna()
        serieses.append(series)

    df = pd.concat(serieses, axis=1)
    result["mean"] = df.mean(axis=1)
    result["mv"] = result["mean"].rolling(window, min_periods=1).mean()
    result["se"] = (df.var(axis=1) / len(df.columns))**0.5
    result = result.fillna(0)
    result["upper"] = result["mv"] + result["se"]
    result["lower"] = result["mv"] - result["se"]
    result = result.add_prefix(prefix)
    return result


def get_asymptotic_performance(file_pattern, n_window=10, episode=200, column="test/success_rate"):
    asymptotic_performance = []
    file_list = glob.glob(file_pattern)
    for file_path in file_list:
        progress_df = pd.read_csv(file_path, index_col=0)
        values = progress_df[column].values.tolist()[episode - n_window: episode]
        asymptotic_performance.append(np.mean(values))
    return asymptotic_performance


def get_time_to_threshold(file_pattern, threshold, n_window=10, n_episodes=200,
                          column="test/success_rate"):
    # 移動平均を取ったあとにTime2Thresholdを取得
    file_list = glob.glob(file_pattern)
    time_to_thresholds = []
    for file_path in file_list:
        progress_df = pd.read_csv(file_path, index_col=0)
        value_df = progress_df[column][:n_episodes]
        maveraged_df = value_df.rolling(window=n_window, min_periods=5).mean()
        v_list = maveraged_df.values.tolist()
        time_to_threshold = len(v_list)
        # import pdb; pdb.set_trace()
        for step, row in enumerate(v_list[5:]):
            if row >= threshold:
                time_to_threshold = step + 1
                break
        time_to_thresholds.append(time_to_threshold)
    return time_to_thresholds


def get_jumpstart(dfs: List[pd.DataFrame], n_episodes: int, column: str) -> List[float]:
    jumpstarts = []

    for df in dfs:
        target_df = df.dropna()
        jumpstarts.append(np.mean(target_df.loc[:n_episodes, column].values))

    return jumpstarts


def main():
    with open("config.json", "r") as f:
        configs = json.load(f)

    t2thres_2, t2thres_4, t2thres_6, t2thres_8, t2thres_9 = {}, {}, {}, {}, {}
    asym_perf, jumpstart = {}, {}
    learning_curves = []

    for file_pattern, prefix in zip(configs["file_patterns"], configs["prefixes"]):
        logger.info("Loading...\n {}".format(file_pattern))
        dfs = read_files(file_pattern)
        jumpstart[prefix] = get_jumpstart(dfs, 100, configs["column"])
        learning_curves.append(
            result_learning_curves(file_pattern, prefix, configs["column"])
        )
        t2thres_2[prefix] = get_time_to_threshold(
            file_pattern, 0.2, column=configs["column"]
        )
        t2thres_4[prefix] = get_time_to_threshold(
            file_pattern, 0.4, column=configs["column"]
        )
        t2thres_6[prefix] = get_time_to_threshold(
            file_pattern, 0.6, column=configs["column"]
        )
        t2thres_8[prefix] = get_time_to_threshold(
            file_pattern, 0.8, column=configs["column"]
        )
        t2thres_9[prefix] = get_time_to_threshold(
            file_pattern, 0.9, column=configs["column"]
        )
        asym_perf[prefix] = get_asymptotic_performance(
            file_pattern, n_window=10 ,episode=200, column=configs["column"]
        )
    learning_curve_df = pd.concat(learning_curves, axis=1)
    learning_curve_df = learning_curve_df.fillna(0)
    out_dir = 'out'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Exporting...")
    learning_curve_df.to_csv(os.path.join(out_dir, "learning_curve.csv"))
    export(os.path.join(out_dir, "time_to_threshold_2.csv"), t2thres_2)
    export(os.path.join(out_dir, "time_to_threshold_4.csv"), t2thres_4)
    export(os.path.join(out_dir, "time_to_threshold_6.csv"), t2thres_6)
    export(os.path.join(out_dir, "time_to_threshold_8.csv"), t2thres_8)
    export(os.path.join(out_dir, "time_to_threshold_9.csv"), t2thres_9)
    export(os.path.join(out_dir, "asymptotic_performance.csv"), asym_perf)
    export(os.path.join(out_dir, "jumpstart.csv"), jumpstart)


if __name__ == "__main__":
    main()
