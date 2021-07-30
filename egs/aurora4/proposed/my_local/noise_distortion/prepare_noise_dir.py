import os
import tqdm
import argparse
import subprocess
from shutil import copyfile


def dump_files(src_dir, out_dir, data):
    with open(f"{out_dir}/wav.scp", "w") as file:
        for key in sorted(data):
            file.write(f"{key} {data[key]}\n")
    copyfile(f"data/{src_dir}/utt2spk", f"{out_dir}/utt2spk")
    copyfile(f"data/{src_dir}/spk2utt", f"{out_dir}/spk2utt")


def prepare_train(kind):
    data_dir = f"data/train_si84_{kind}_sp_hires"
    os.makedirs(f"{data_dir}/wav", exist_ok=True)
    os.makedirs(f"{data_dir}/log", exist_ok=True)

    clean_set = "train_si84_clean_sp_hires"
    multi_set = "train_si84_multi_sp_hires"

    data = dict()

    with open(f"data/{multi_set}/wav.scp") as multi_file, open(f"data/{clean_set}/wav.scp") as clean_file,\
        open(f"data/train_si84_{kind}_sp_hires/log/mix_wav.log", "w") as log_file:
        for multi_line, clean_line in tqdm.tqdm(zip(multi_file, clean_file)):
            multi_key, multi_cmd = multi_line.strip().split(" ", 1)
            clean_key, clean_cmd = clean_line.strip().split(" ", 1)
            multi_cmd = multi_cmd[:-2]
            clean_cmd = clean_cmd[:-2]

            # compute the difference between 2 wav files and output a new wav
            cmd = f"sox -m -v 1 '|{multi_cmd}' -v -1 '|{clean_cmd}' {data_dir}/wav/{multi_key}.wav"
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            data[multi_key] = f"{data_dir}/wav/{multi_key}.wav" # cmd

            log_file.write(f"{multi_key} {cmd}\n")
            
    # Dump wav.scp, utt2spk, spk2utt
    dump_files(multi_set, data_dir, data)


def prepare_test(kind):
    clean_cmd_dict = dict()

    for data_set in ["test_A_hires", "test_B_hires", "test_C_hires", "test_D_hires"]:
        data_dir = "data/" + data_set.replace("_hires", f"_{kind}_hires")
        os.makedirs(f"{data_dir}/wav", exist_ok=True)
        os.makedirs(f"{data_dir}/log", exist_ok=True)

        data = dict()

        with open(f"data/{data_set}/wav.scp") as multi_file,\
            open(f"{data_dir}/log/mix_wav.log", "w") as log_file:
            for multi_line in tqdm.tqdm(multi_file):
                multi_key, multi_cmd = multi_line.strip().split(" ", 1)
                multi_cmd = multi_cmd[:-2]
                
                if "A" in data_set:
                    clean_cmd_dict[multi_key] = multi_cmd
                    # compute the difference between itself
                    cmd = f"sox -m -v 1 '|{multi_cmd}' -v -1 '|{multi_cmd}' {data_dir}/wav/{multi_key}.wav"
                else:
                    clean_cmd = clean_cmd_dict[multi_key[:-1]+"0"]
                    # compute the difference between 2 wav files and output a new wav
                    cmd = f"sox -m -v 1 '|{multi_cmd}' -v -1 '|{clean_cmd}' {data_dir}/wav/{multi_key}.wav"
                
                p = subprocess.Popen(cmd, shell=True)
                p.communicate()
                data[multi_key] = f"{data_dir}/wav/{multi_key}.wav" # cmd

                log_file.write(f"{multi_key} {cmd}\n")
                
        # Dump wav.scp, utt2spk, spk2utt
        dump_files(data_set, data_dir, data)


def main(kind):
    prepare_train(kind)
    prepare_test(kind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", default="noise_mismatch", nargs="?",
                        help="'noise_mismatch' might contain channel distortion, "\
                             "'pure_noise' only extract added noise (TODO)")
    args = parser.parse_args()

    main(kind=args.kind)