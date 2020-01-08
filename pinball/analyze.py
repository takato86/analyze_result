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


def load_total_reward(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = []
    for file_path in file_list:
        total_reward_df = pd.read_csv(file_path, index_col=0)
        total_reward = total_reward_df.values.reshape(len(total_reward_df.index))
        total_rewards.append(total_reward.tolist())
    return total_rewards


def average_values(file_pattern):
    total_rewards = load_total_reward(file_pattern)
    total_rewards = np.array(total_rewards)
    mean_total_rewards = np.mean(total_rewards, axis=0)
    return mean_total_rewards


def output_standard_error(file_pattern):
    total_rewards = load_total_reward(file_pattern)
    total_rewards = np.array(total_rewards)
    std_total_rewards = np.std(total_rewards, axis=0)
    ste_total_rewards = std_total_rewards / np.sqrt(total_rewards.shape[0])
    return ste_total_rewards


def output_time_to_threshold(file_pattern, threshold):
    total_rewards = load_total_reward(file_pattern)
    time_to_threshold = []
    moved_averaged = calc_moved_average(total_rewards)
    for total_reward in moved_averaged:
        flag = False
        for episode, reward in enumerate(total_reward):
            if threshold < reward:
                time_to_threshold.append(episode)
                flag = True
                break
        if episode == len(total_reward) - 1 and not flag:
            time_to_threshold.append(episode)
    return time_to_threshold


def output_asymptotic_performance(file_pattern, window=10):
    total_rewards = load_total_reward(file_pattern)
    asymptotic_performances = []
    for total_reward in total_rewards:
        t_reward = np.array(total_reward[len(total_reward)-window:])
        asymptotic_performances.append(np.mean(total_reward))
    return asymptotic_performances


def calc_moved_average(total_rewards, window=10):
    moved_average_total_rewards = []
    mean_total_rewards = np.mean(total_rewards, axis=0)
    for total_reward in total_rewards:
        moved_average_total_reward = []
        for episode, reward in enumerate(total_reward):
            if episode < window:
                moved_average = mean_total_rewards[episode]
            else:
                moved_average\
                    = sum(total_reward[episode - window: episode+1]) / window
            moved_average_total_reward.append(moved_average)
        moved_average_total_rewards.append(moved_average_total_reward)
    return moved_average_total_rewards


def main():
    argvs = sys.argv[1:]
    processed_v = {}
    time_to_threshold = {}
    asymptotic_performances = {}
    thresholds = [0, 1000, 2000, 4000, 6000, 7000, 8000, 9000]
    for argv in argvs:
        processed_v[argv+"_"+"mean"] = average_values(argv)
        processed_v[argv+"_"+"ste"] = output_standard_error(argv)
        asymptotic_performances[argv] = output_asymptotic_performance(argv)
        for threshold in thresholds:
            time_to_threshold[str(threshold) + "_" + argv]\
                = output_time_to_threshold(argv, threshold)
    out_dir = 'out'
    file_name = 'mean_ste_total_reward.csv'
    file_path = os.path.join(out_dir, file_name)
    export(file_path, processed_v)

    file_name = 'time_to_threshold_total_reward.csv'
    file_path = os.path.join(out_dir, file_name)
    export(file_path, time_to_threshold)

    file_name = 'asymptotic_performance_total_reward.csv'
    file_path = os.path.join(out_dir, file_name)
    export(file_path, asymptotic_performances)




if __name__ == "__main__":
    main()