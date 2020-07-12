
import pandas as pd
import os
import re
import pickle
import random

def _split_train_val():
    if not os.path.exists("./netflix/train"):
        os.mkdir("./netflix/train")
    if not os.path.exists("./netflix/val"):
        os.mkdir("./netflix/val")

    prob_file_path = "./netflix/probe.txt"
    user_id = set()
    finished_files = []
    with open(prob_file_path, "r") as probe_file:
        lines = probe_file.read()
        for i, s in zip(re.findall(r"[0-9]+:\n", lines), re.compile(r"[0-9]+:\n").split(lines)[1:]):
            data_file = "./netflix/training_set/" + "mv_{:07d}.txt".format(int(i[:-2]))
            csv_original = pd.read_csv(data_file, header=None, skiprows=[0])
            cur_user_id = set(csv_original.iloc[:, 0].unique().flatten())
            user_id = user_id.union(cur_user_id)
            csv_val_index = [int(num) for num in s[:-1].split("\n")]
            csv_val = csv_original.loc[csv_original.iloc[:, 0].isin(csv_val_index)]
            csv_val.insert(0, None, int(i[:-2]))
            csv_val.to_csv("./netflix/val/" + "mv_{:07d}.txt".format(int(i[:-2])), header=False, index=False)
            csv_train = csv_original.loc[~csv_original.iloc[:, 0].isin(csv_val_index)]
            csv_train.insert(0, None, int(i[:-2]))
            csv_train.to_csv("./netflix/train/" + "mv_{:07d}.txt".format(int(i[:-2])), header=False, index=False)
            finished_files.append(int(i[:-2]))
            if len(finished_files) % 100 == 0:
                print("FINISH {}.".format(len(finished_files)))

    for i in range(1, 17771):
        if i not in finished_files:
            data_file = "./netflix/training_set/" + "mv_{:07d}.txt".format(i)
            csv_original = pd.read_csv(data_file, header=None, skiprows=[0])
            cur_user_id = set(csv_original.iloc[:, 0].unique().flatten())
            user_id = user_id.union(cur_user_id)
            csv_original.insert(0, None, i)
            csv_original.to_csv("./netflix/train/" + "mv_{:07d}.txt".format(i), header=False, index=False)
            finished_files.append(i)
            if len(finished_files) % 100 == 0:
                print("FINISH {}.".format(len(finished_files)))

    pickle.dump(user_id, open("./netflix/set.pickle", "wb"))
    print("TOTAL USERS: {}.".format(len(user_id)))
    print("TOTAL MOVIES: {}.".format(len(finished_files)))

def _shuffle_and_split(num_files=50, filepath="./netflix/train.txt", dump_path="./train", prefix="train"):
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    lines = open(filepath, "rb").readlines()
    print("TOTAL LINES: {}.".format(len(lines)))

    random.shuffle(lines)
    each = len(lines) // num_files
    res = len(lines) % num_files

    start = 0
    end = 0
    for i in range(res):
        end += each + 1
        open(os.path.join(dump_path, "{}_{}.txt".format(prefix, i+1)), "wb").writelines(lines[start: end])
        start = end
        print("ALREADY PROCESSED {}.".format(start))

    for i in range(res, num_files):
        end += each
        open(os.path.join(dump_path, "{}_{}.txt".format(prefix, i+1)), "wb").writelines(lines[start: end])
        start = end
        print("ALREADY PROCESSED {}.".format(start))

    print("TOTAL RECORDS: {}.".format(start))

if __name__ =="__main__":
    # _split_train_val()
    # user_id = pickle.load(open("./netflix/set.pickle", "rb"))
    # user_dict = {user:i for i, user in enumerate(list(user_id))}
    # pickle.dump(user_dict, open("./netflix/dict.pickle", "wb"))
    _shuffle_and_split()
