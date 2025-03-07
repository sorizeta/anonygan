import os
import shutil
import pandas as pd
import itertools
import random
import torch
import glob
from landmark_detection import *

# Functions to load samples from folders

def path_generate_train_pairs(dest_dir):

    df = pd.DataFrame()

    images_list = glob.glob(dest_dir + '/*.jpg')
    images_list = list(map(lambda x: str(x.split("/")[-1]), images_list))
    images_list = list(map(lambda x: str(x.split(".")[0]), images_list))
    all_couples = list(itertools.combinations(images_list, 2))
    df = df.append(all_couples)

    df.columns = ['from', 'to']

    df.to_csv(os.path.join(dest_dir, "celeba-pairs-train.csv"), index=False, mode="w")


def detect_landmarks(root_dir, save_name):
    files = glob.glob(root_dir + '/*.jpg')
    lnd_list = []
    idx = ["{}".format(x) for x in range(136)]
    

    for f in files:
        path = os.path.join(root_dir, f)
        _, lnd = read_im_and_landmarks(path)
        lnd = np.array(lnd).reshape(-1,).tolist()
        lnd_dict = dict(zip(idx, lnd))
        if len(lnd) > 0:
            filename = str(f.split("/")[-1])
            lnd_dict['filename'] = str(filename)
            lnd_list.append(lnd_dict)
    
    df = pd.DataFrame(lnd_list)
    file_path = os.path.join(root_dir, save_name)
    df.to_csv(file_path, sep='\t', index=False)
    return df


def path_generate_landmarks(dest_dir, landmark_root_dir, landmark_file):

    df = pd.read_csv(os.path.join(dest_dir, "celeba-pairs-train.csv"), dtype=str, sep=',')
    all_values = df.to_numpy().flatten().tolist()
    all_values = list(map(lambda x: f"{x}.jpg", all_values))

    landmarks_df: pd.DataFrame = pd.read_csv(os.path.join(dest_dir, landmark_file), header=0, sep='\t')
    
    reduced_lndmks = landmarks_df[landmarks_df['filename'].isin(all_values)]
    
    # for 29-kp dataset
    #legal_lndmk = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 27,28,29,30, 60,61,62,63,64,65,66,67]
    # for full 68 kp
    legal_lndmk = list(range(68))

    def create_image(row):
        name = str(row['filename'])
        if not os.path.exists(os.path.join(landmark_root_dir, name)):
            lndmks = row[:-1].tolist()
            a = torch.zeros((len(legal_lndmk), 218, 178), dtype=torch.int8)
            try:
                for i in range(len(legal_lndmk)):
                    x, y = lndmks[2* legal_lndmk[i]], lndmks[2*legal_lndmk[i] + 1]
                    x = int(x)
                    y = int(y)
                    a[i][y][x] = 1

                name = name.split('.')[0]
                torch.save(a, os.path.join(landmark_root_dir, f"{name}.pt"))
                print(f"Done {name}")
            except Exception as e:
                print(f"Problems with {name}", e)

    reduced_lndmks.apply(lambda x: create_image(x), axis=1)


######################
# Original functions
#####################

# Keeping them as a guide

def select_100_train():
    # move 100 random ids w/ at least 25 images
    root_dir = "/media/dvl1/SSD_DATA/CelebA/poses_4_hao"
    dest_dir = "/media/dvl1/SSD_DATA/BiGraphCeleba/train"

    train_idx = []
    
    idx = 0
    for fld in os.listdir(root_dir):
        src = os.path.join(root_dir, fld)
        dst = os.path.join(dest_dir, fld)
        
        # 14 is the test conditioning image
        if fld != "14":
            # shutil.copytree(src, dst)
            train_idx.append(fld)
            idx += 1
        
        if idx > 99:
            print("Done...")
            return 


def select_100_test():
    # test set
    root_dir = "/media/dvl1/SSD_DATA/CelebA/poses_4_hao"
    dest_dir = "/media/dvl1/SSD_DATA/BiGraphCeleba/test"

    train_idx = select_100_train()
    test_idx = []
    
    idx = 0
    for fld in os.listdir(root_dir):
        
        src = os.path.join(root_dir, fld)
        dst = os.path.join(dest_dir, fld)

        if fld != "14" and fld not in train_idx:
            # print(fld)
            # shutil.copytree(src, dst)
            test_idx.append(fld)
            idx += 1
        
        if idx > 99:
            print("Done...")
            return test_idx


def generate_train_pairs():
    # extract random pairs of images, 100 per id
    dest_dir = "/media/dvl1/SSD_DATA/BiGraphCeleba/train"
    df = pd.DataFrame()

    for _id in os.listdir(dest_dir):
        images_list = os.listdir(os.path.join(dest_dir, _id))
        images_list = list(map(lambda x: str(x.split(".")[0]), images_list))
        all_couples = list(itertools.combinations(images_list, 2))
        subset = random.sample(all_couples, 100)
        df = df.append(subset)

    df.columns = ['from', 'to']

    df.to_csv(os.path.join(dest_dir, "celeba-pairs-train.csv"), index=False, mode="w")


def extract_landmarks():
    # for the used images, extract landmarks and save the in suitable format
    dest_dir = "/media/dvl1/SSD_DATA/bigraph-dataset/"
    df = pd.read_csv(os.path.join(dest_dir, "celeba-pairs-train.csv"), dtype=str)
    all_values = df.to_numpy().flatten().tolist()
    all_values = list(map(lambda x: f"{x}.jpg", all_values))

    landmarks_df: pd.DataFrame = pd.read_csv("/media/dvl1/SSD_DATA/CelebA/celeba_landmarks/landmark_align.txt", header=None, delim_whitespace=True)

    reduced_lndmks = landmarks_df[landmarks_df[0].isin(all_values)]

    root_dir = "/media/dvl1/SSD_DATA/bigraph-dataset/trainK_68"

    # for 29-kp dataset
    #legal_lndmk = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 27,28,29,30, 60,61,62,63,64,65,66,67]
    # for full 68 kp
    legal_lndmk = list(range(68))

    def create_image(row):
        name = str(row[0])
        if not os.path.exists(os.path.join(root_dir, name)):
            lndmks = row[1:].tolist()
            
            a = torch.zeros((len(legal_lndmk), 218, 178), dtype=torch.int8)
            try:
                for i in range(len(legal_lndmk)):
                    x, y = lndmks[2* legal_lndmk[i]], lndmks[2*legal_lndmk[i] + 1]
                    a[i][y][x] = 1

                name = name.split('.')[0]
                torch.save(a, os.path.join(root_dir, f"{name}.pt"))
                print(f"Done {name}")
            except Exception as e:
                print(f"Problems with {name}", e)

    reduced_lndmks.apply(lambda x: create_image(x), axis=1)