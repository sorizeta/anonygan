import shutil
import pandas as pd
import numpy as np
import sys, os
import argparse


def select_identities(id_path, source_path, dest_path, number_ids = 15):
    id_df = pd.read_csv(id_path, header=None, names=['filename', 'identity'], delim_whitespace=True)
    ids = id_df['identity']
    sampled_ids = np.random.choice(ids.unique(), size=number_ids)
    sample_df = id_df.loc[id_df['identity'].isin(sampled_ids)]
    for row in sample_df.itertuples():
        f = row[1]
        identity = row[2]
        folder_path = os.path.join(dest_path, identity)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        src_file = os.path.join(source_path, f)
        dest_file = os.path.join(folder_path,  f)
        shutil.copyfile(src_file, dest_file)
    return


def select_single_identities(id_path, source_path, dest_path, number_ids = 15):
    id_df = pd.read_csv(id_path, header=None, names=['filename', 'identity'], delim_whitespace=True)
    ids = id_df['identity']
    sampled_ids = np.random.choice(ids.unique(), size=number_ids)
    sample_df = id_df.loc[id_df['identity'].isin(sampled_ids)]
    filenames = list(sample_df.groupby('identity').apply(lambda x: x.sample(1)).reset_index(drop=True))
    for f in filenames:
        src_file = os.path.join(source_path, f)
        dest_file = os.path.join(dest_path,  f)
        shutil.copyfile(src_file, dest_file)
    return


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Samples CelebA images based on identities')
        parser.add_argument('id_file', type=str, help='Path of the identity file')
        parser.add_argument('src', type=str, help='Images folder')
        parser.add_argument('dest', type=str, help='Destination folder')
        parser.add_argument('--number', type=int, help='Number of identities to retrieve (default 15')
        parser.add_argument('--distinct', action='store_true', help='Retrieves a single image for each id')

        args = parser.parse_args()

        if args.number is None:
            args.number = 15
        
        if args.distinct:
            select_single_identities(args.id_file, args.src, args.dest, args.number)
        else:
            select_identities(args.id_file, args.src, args.dest, args.number)

    except Exception as e:
        print(e)
    
