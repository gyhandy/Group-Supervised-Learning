import os
import random
import shutil


Raw_Data_root = '/home2/RaFD/data'

included_ep = ['happy', 'neutral', 'surprised', 'disgusted', 'contemptuous']

test_flag = 0
for roots, dirs, files in os.walk(Raw_Data_root):
    if files:
        total = len(files)
        for i, file in enumerate(files):
            test_flag = random.randint(1, 10)
            if test_flag > 8:
                test_flag = 1
            else:
                test_flag = 0
            print(i, '/', total)
            file_path = os.path.join(roots, file)
            fl = file.split('_')
            iden = fl[1]
            pz = fl[0][4:]
            ep = fl[4]
            gaze = fl[5]
            if ep not in included_ep or gaze != 'frontal.jpg':
                # print(gaze)
                continue
            if test_flag == 1:
                id_dir = os.path.join('/home2/RaFD/train/img_id', iden)
                pz_dir = os.path.join('/home2/RaFD/train/img_pz', pz)
                ep_dir = os.path.join('/home2/RaFD/train/img_ep', ep)
            elif test_flag == 0:
                id_dir = os.path.join('/home2/RaFD/test/img_id', iden)
                pz_dir = os.path.join('/home2/RaFD/test/img_pz', pz)
                ep_dir = os.path.join('/home2/RaFD/test/img_ep', ep)
            if not os.path.exists(id_dir):
                os.makedirs(id_dir)
            if not os.path.exists(pz_dir):
                os.makedirs(pz_dir)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            id_path = os.path.join(id_dir, iden + '_' + pz + '_' + ep + '.jpg')
            pz_path = os.path.join(pz_dir, iden + '_' + pz + '_' + ep + '.jpg')
            ep_path = os.path.join(ep_dir, iden + '_' + pz + '_' + ep + '.jpg')
            shutil.copy(file_path, id_path)
            shutil.copy(file_path, pz_path)
            shutil.copy(file_path, ep_path)
            if test_flag == 1:
                shutil.copy(file_path, os.path.join('/home2/RaFD/train/data', iden + '_' + pz + '_' + ep + '.jpg'))
            elif test_flag == 0:
                shutil.copy(file_path, os.path.join('/home2/RaFD/test/data', iden + '_' + pz + '_' + ep + '.jpg'))