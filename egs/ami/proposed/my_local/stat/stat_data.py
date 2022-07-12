import os
import argparse


def stat_duration(data_dir):
    utt2dur_filename = f"{data_dir}/utt2dur"
    if os.path.exists(utt2dur_filename):
        duration = 0
        with open(utt2dur_filename, "r") as file:
            for line in file:
                uid, dur = line.strip().split(" ", 1)
                duration += float(dur)
        print("Duration (sec):", duration)
        print("Duration (hr):", duration/3600)


def stat_complete_duration(data_dir):
    import re
    import librosa
    
    wavscp_filename = f"{data_dir}/wav.scp"
    if os.path.exists(wavscp_filename):
        duration = 0
        with open(wavscp_filename, "r") as file:
            for line in file:
                matched = re.search("/mnt.+?wav", line)
                filename = matched.group(0)
                duration += librosa.get_duration(filename=filename)
        print("Complete Duration (sec):", duration)
        print("Complete Duration (hr):", duration/3600)


def stat_utt(data_dir):
    utt2dur_filename = f"{data_dir}/utt2dur"
    if os.path.exists(utt2dur_filename):
        number = 0
        with open(utt2dur_filename, "r") as file:
            for line in file:
                number += 1
        print("Utterence :", number)
        

def stat_speaker(data_dir):
    spk2utt_filename = f"{data_dir}/spk2utt"
    if os.path.exists(spk2utt_filename):
        number = 0
        with open(spk2utt_filename, "r") as file:
            for line in file:
                number += 1
        print("Speaker number:", number)


def stat_rec(data_dir):
    wavscp_filename = f"{data_dir}/wav.scp"
    if os.path.exists(wavscp_filename):
        number = 0
        with open(wavscp_filename, "r") as file:
            for line in file:
                number += 1
        print("Recording number:", number)


def stat_gender(data_dir):
    spk2gender_filename = f"{data_dir}/spk2gender"
    if os.path.exists(spk2gender_filename):
        male, female = 0, 0
        with open(spk2gender_filename, "r") as file:
            for line in file:
                spk, gender = line.strip().split(" ", 1)
                if gender == "m": male += 1
                elif gender == "f": female += 1
        print("Male number:", male)
        print("Female number:", female)


def stat_meeting(data_dir):
    wavscp_filename = f"{data_dir}/wav.scp"
    if os.path.exists(wavscp_filename):
        meeting_id_set = set()
        with open(wavscp_filename, "r") as file:
            for line in file:
                meeting_id_set.add(line.split(" ")[0].rsplit("_", 1)[0])
        print("Meeting number:", len(meeting_id_set))


def main(data_dir):
    print(f"Stating {data_dir}...")
    stat_duration(data_dir)
    # stat_complete_duration(data_dir)
    stat_utt(data_dir)
    # stat_speaker(data_dir)
    stat_rec(data_dir)
    stat_gender(data_dir)
    stat_meeting(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", default="data/sdm1/train", nargs="?")
    args = parser.parse_args()

    main(args.data_dir)





