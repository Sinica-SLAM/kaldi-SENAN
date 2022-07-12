import os
import tqdm
import argparse
import subprocess
from shutil import copyfile


def main(mic, kind):
    data_dir = f"data/{mic}/train_cleaned_{kind}_sp_hires_40"
    os.makedirs(f"{data_dir}/wav", exist_ok=True)
    os.makedirs(f"{data_dir}/log", exist_ok=True)

    clean_set = "train_cleaned_ihmdata_sp_hires_40"
    multi_set = "train_cleaned_sp_hires_40"

    data = dict()

    with open(f"data/{mic}/{multi_set}/wav.scp") as multi_file, \
        open(f"data/{mic}/train_cleaned_{kind}_sp_hires_40/log/mix_wav.log", "w") as log_file:
        for multi_line in tqdm.tqdm(multi_file):
            multi_key, multi_cmd = multi_line.strip().split(" ", 1)
            _multi_key = multi_key.rsplit("_", 1)[0]
            multi_cmd = multi_cmd[:-2]
            
            with open(f"/mnt/HDD2/user_pinyuanc/mod-kaldi/kaldi-slam/egs/ami/proposed/data/{mic}/{clean_set}/wav.scp") as clean_file:
                for clean_line in clean_file:
                    clean_key, clean_cmd = clean_line.strip().split(" ", 1)
                    _clean_key = clean_key.rsplit("_", 1)[0]
                    clean_cmd = clean_cmd[:-2]

                    if _multi_key == _clean_key: 
                        # compute the difference between 2 wav files and output a new wav
                        # Note: speed could be consistent, but volume not in this case!
                        cmd = f"sox -m -v 1 '|{multi_cmd}' -v -1 '|{clean_cmd}' {data_dir}/wav/{multi_key}.wav"
                        cmd = f"sox -m -v 1 '|{multi_cmd}' -v -1 '|{clean_cmd}' {data_dir}/wav/{multi_key}.wav"
                        p = subprocess.Popen(cmd, shell=True)
                        p.communicate()
                        data[multi_key] = f"{data_dir}/wav/{multi_key}.wav" # cmd

                        log_file.write(f"{multi_key} {cmd}\n")
                        break
            
    # Dump wav.scp, utt2spk, spk2utt, segments
    with open(f"{data_dir}/wav.scp", "w") as file:
        for key in sorted(data):
            file.write(f"{key} {data[key]}\n")
    copyfile(f"data/{mic}/{multi_set}/utt2spk", f"{data_dir}/utt2spk")
    copyfile(f"data/{mic}/{multi_set}/spk2utt", f"{data_dir}/spk2utt")
    copyfile(f"data/{mic}/{multi_set}/segments", f"{data_dir}/segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mic", default="sdm1", nargs="?")
    parser.add_argument("kind", default="noise_mismatch", nargs="?",
                        help="'noise_mismatch' might contain channel distortion")
    args = parser.parse_args()

    main(mic=args.mic, kind=args.kind)
