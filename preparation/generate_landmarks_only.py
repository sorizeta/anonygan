from preparation.train_images_and_keypoints_1 import path_generate_train_pairs
from train_images_and_keypoints_1 import *
import os

images_path = '/home/ubuntu/anonygan-prova'
landmark_file_name = 'test_landmarks.csv'
landmark_folder = os.path.join(images_path, 'landmarks')

path_generate_train_pairs(images_path)
detect_landmarks(images_path, landmark_file_name)
path_generate_landmarks(images_path, landmark_folder, landmark_file_name)