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
    export_df = pd.DataFrame(content)
    export_df.to_csv(out_file_path)
    logger.info(f"Export {out_file_path}")


def average_values(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = []
    for file_path in file_list:
        total_reward_df = pd.read_csv(file_path, index_col=0)
        total_reward = total_reward_df.values.reshape(len(total_reward_df.index))
        total_rewards.append(total_reward.tolist())
    total_rewards = np.array(total_rewards)
    mean_total_rewards = np.mean(total_rewards, axis=0)
    return mean_total_rewards


def output_standard_error(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = []
    for file_path in file_list:
        total_reward_df = pd.read_csv(file_path, index_col=0)
        total_reward = total_reward_df.values.reshape(len(total_reward_df.index))
        total_rewards.append(total_reward.tolist())
    total_rewards = np.array(total_rewards)
    std_total_rewards = np.std(total_rewards, axis=0)
    ste_total_rewards = std_total_rewards / np.sqrt(total_rewards.shape[0])
    return ste_total_rewards


def main():
    argvs = sys.argv[1:]
    processed_v = {}
    for argv in argvs:
        processed_v[argv+"_"+"mean"] = average_values(argv)
        processed_v[argv+"_"+"ste"] = output_standard_error(argv)
    out_dir = 'out'
    file_name = 'mean_ste_total_reward.csv'
    file_path = os.path.join(out_dir, file_name)
    export(file_path, processed_v)


if __name__ == "__main__":
    main()