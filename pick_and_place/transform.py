"""tensorboard出力ファイルからanalyze.pyで利用できるcsvに変換するモジュール."""
import tbdf
import os
import glob
import logging
from tqdm import tqdm


def transform_csv(file_patterns):
    logging.info("START")
    file_list = []
    
    for fp in file_patterns:
        file_list += glob.glob(fp)
    
    for fpath in tqdm(file_list):
        dirname = os.path.dirname(fpath)
        df = tbdf.load_as_dataframe(fpath)
        df.to_csv(os.path.join(dirname, "progress.csv"))

    logging.info("FINISH")


if __name__ == "__main__":
    file_patterns = [
        "/home/tokudo/Develop/research/pdrl/runs/train/baseline/*/events.out.tfevents.*",
        "/home/tokudo/Develop/research/pdrl/runs/train/baseline/*/events.out.tfevents.*",
    ]
    transform_csv(file_patterns)