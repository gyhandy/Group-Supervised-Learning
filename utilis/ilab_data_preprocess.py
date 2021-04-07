'''
iLab dataset is organized by pose originaly.
To help the Group supervised learning training on ilab20M
This script reorganize the ilab-20M dataset with identity and background

'''
import os
import shutil


'''
identity
'''
Raw_Data_root = '/home2/ilab2M_pose/train_img_c00_10class'
Target_root ='/home2/ilab2M_pose/vae_identity_new/'
pose_info = ['c00_r01', 'c00_r02', 'c00_r03', 'c00_r04', 'c00_r06', 'c00_r07']
for roots, dirs, files in os.walk(Raw_Data_root):
    for i, file in enumerate(files):
        category = file.split('-')[0]
        identity = file.split('-')[1]
        A_path = os.path.join(roots, file)
        if not os.path.exists(os.path.join(Target_root, category, identity)): # category identity
            os.makedirs(os.path.join(Target_root, category, identity))
        shutil.copy(os.path.join(roots, file), os.path.join(Target_root, category, identity, file))

'''
background
'''

Raw_Data_root = '/home2/ilab2M_pose/train_img_c00_10class'
Target_root ='/home2/ilab2M_pose/vae_back_new/'
pose_info = ['c00_r01', 'c00_r02', 'c00_r03', 'c00_r04', 'c00_r06', 'c00_r07']
for roots, dirs, files in os.walk(Raw_Data_root):
    for i, file in enumerate(files):
        category = file.split('-')[0]
        pose = file.split('-')[3] + '_' + file.split('-')[4]
        back = file.split('-')[2]
        A_path = os.path.join(roots, file)
        if not os.path.exists(os.path.join(Target_root, back, category, pose)): # back info
            os.makedirs(os.path.join(Target_root, back, category, pose))
        shutil.copy(os.path.join(roots, file), os.path.join(Target_root, back, category, pose, file))



