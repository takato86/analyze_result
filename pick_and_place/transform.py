"""tensorboard出力ファイルからanalyze.pyで利用できるcsvに変換するモジュール."""
import tbdf
import os
import logging
from tqdm import tqdm


def transform_csv(file_paths):
    logging.info("START")
    for fpath in tqdm(file_paths):
        dirname = os.path.dirname(fpath)
        df = tbdf.load_as_dataframe(fpath)
        df.to_csv(os.path.join(dirname, "progress.csv"))

    logging.info("FINISH")


if __name__ == "__main__":
    file_list = [
        "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-06-07 21_33_48.795040_0/events.out.tfevents.1654605229.ymdlab00.11191.0",
        "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-06-08 03_22_00.048305_0/events.out.tfevents.1654626121.ymdlab00.24512.0"
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-06-07 13_57_05.972482_0/events.out.tfevents.1654577827.ymdlab00.11698.0"
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-05-16 16_53_27.205090_0/events.out.tfevents.1652687609.ymdlab00.23384.0",
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-05-24 14_03_17.102686_0/events.out.tfevents.1653368598.ymdlab00.15147.0",
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-05-26 16_15_58.003699_0/events.out.tfevents.1653549358.ymdlab00.11062.0",
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-05-28 23_25_07.439907_0/events.out.tfevents.1653747908.ymdlab00.30045.0",
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-05-30 15_02_14.781031_0/events.out.tfevents.1653890536.ymdlab00.21135.0",
        # "/home/tokudo/develop/analyze_result/pick_and_place/train/2022-06-05 22_41_30.880591_0/events.out.tfevents.1654436492.ymdlab00.18405.0"
    ]
    transform_csv(file_list)