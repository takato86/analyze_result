import json
from typing import List
import pandas as pd
import numpy as np
import glob
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_file_list(file_pattern):
    return glob.glob(file_pattern)


def export(out_file_path, content):
    l = max(map(lambda x: len(x[1]), content.items()))
    filled_content = {}
    try:
        for key, val in content.items():
            filled_content[key] = list(val) + [None]*(l-len(val))
    except ValueError:
        import pdb; pdb.set_trace()
    export_df = pd.DataFrame(filled_content)
    export_df.to_csv(out_file_path)
    logger.info(f"Export {out_file_path}")


def load_total_reward(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = []
    for file_path in file_list:
        logger.info(f"Processing on {file_path} ....")
        total_reward_df = pd.read_csv(file_path, index_col=0)
        total_reward = total_reward_df.values.reshape(len(total_reward_df.index))
        total_rewards.append(total_reward.tolist())
    return total_rewards


def average_values(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = load_total_reward(file_pattern)
    total_rewards = np.array(total_rewards)
    mean_total_rewards = np.mean(total_rewards, axis=0)
    var_total_rewards = np.var(total_rewards, axis=0)
    standard_error = (var_total_rewards / len(file_list))**0.5
    return mean_total_rewards, standard_error


def get_asymptotic_performance(file_pattern, n_window=10, episode=200):
    asymptotic_performance = []
    file_list = get_file_list(file_pattern)
    for file_path in file_list:
        value_df = pd.read_csv(file_path, index_col=0)
        values = value_df.values.tolist()[episode - n_window: episode]
        asymptotic_performance.append(np.mean(values))
    return asymptotic_performance


def get_time_to_threshold(file_pattern, threshold, n_window=10):
    # 移動平均を取ったあとにTime2Thresholdを取得
    file_list = get_file_list(file_pattern)
    time_to_thresholds = []

    for file_path in file_list:
        value_df = pd.read_csv(file_path, index_col=0)
        maveraged_df = value_df.rolling(window=n_window, min_periods=5).mean()
        v_list = maveraged_df.values.tolist()
        time_to_threshold = len(v_list)
        for step, row in enumerate(v_list[5:]):
            if row[0] <= threshold:
                time_to_threshold = step + 1
                break
        time_to_thresholds.append(time_to_threshold)
    return time_to_thresholds


def get_jumpstart(dfs: List[pd.DataFrame], n_episodes: int):
    jumpstarts = []

    for df in dfs:
        jumpstarts.append(np.mean(df[:n_episodes].values))

    return jumpstarts


def read_files(file_pattern: str) -> List[pd.DataFrame]:
    dfs = []

    for fname in get_file_list(file_pattern):
        dfs.append(pd.read_csv(fname, index_col=0))

    return dfs


def main():
    with open("config.json", "r") as f:
        configs = json.load(f)

    file_patterns = configs["file_patterns"]
    prefixes = configs["prefixes"]
    asym_perf, learning_curve_dict, jumpstart_dict = {}, {}, {}
    t2thres_500, t2thres_1000, t2thres_2000, t2thres_3000 = {}, {}, {}, {}

    for prefix, file_pattern in zip(prefixes, file_patterns):
        logger.info("Loading...\n {}".format(file_pattern))
        dfs = read_files(file_pattern)
        jumpstart_dict[prefix] = get_jumpstart(dfs, 1)
        # TODO dfsを使うように修正。
        learning_curve_dict[f"{prefix}-mean"], learning_curve_dict[f"{prefix}-se"] = average_values(file_pattern)
        learning_curve_dict[f"{prefix}-mv"] = pd.Series(learning_curve_dict[f"{prefix}-mean"]).rolling(10, min_periods=1).mean()
        learning_curve_dict[f"{prefix}-lower"] = learning_curve_dict[f"{prefix}-mv"] - learning_curve_dict[f"{prefix}-se"]
        learning_curve_dict[f"{prefix}-upper"] = learning_curve_dict[f"{prefix}-mv"] + learning_curve_dict[f"{prefix}-se"]
        t2thres_3000[prefix] = get_time_to_threshold(file_pattern, 3000)
        t2thres_2000[prefix] = get_time_to_threshold(file_pattern, 2000)
        t2thres_1000[prefix] = get_time_to_threshold(file_pattern, 1000)
        t2thres_500[prefix] = get_time_to_threshold(file_pattern, 500)
        asym_perf[prefix] = get_asymptotic_performance(file_pattern, n_window=1 ,episode=250)

    out_dir = 'out'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Exporting...")
    export(os.path.join(out_dir, "learning_curve.csv"), learning_curve_dict)
    export(os.path.join(out_dir, "time_to_threshold_3000.csv"), t2thres_3000)
    export(os.path.join(out_dir, "time_to_threshold_500.csv"), t2thres_500)
    export(os.path.join(out_dir, "time_to_threshold_2000.csv"), t2thres_2000)
    export(os.path.join(out_dir, "time_to_threshold_1000.csv"), t2thres_1000)
    export(os.path.join(out_dir, "asymptotic_performance.csv"), asym_perf)
    export(os.path.join(out_dir, "jumpstart.csv"), jumpstart_dict)

if __name__ == "__main__":
    main()