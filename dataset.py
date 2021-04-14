"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import transforms as T
from image_folder import make_dataset, group_path
from PIL import Image
from PIL import ImageFile
import random
import fnmatch

ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class ilab_imgfolder(Dataset):
    def __init__(self, root, transform=None, train=True):
        super(ilab_imgfolder, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        if self.train:
            self.paths = make_dataset(self.root)
            self.C_size = len(self.paths) # size of center image C
        else: # test mode
            self.C_size, self.paths = group_path(self.root) # size of center image C

    def findABD(self, index):
        '''
        refer paper Fig.3
        C: x
        A: same pose as C
        B: same identity as C
        D: same background as C
        '''
        FOUNDED = False
        while not FOUNDED:
            FOUNDED = True
            C_img_path = self.paths[index % self.C_size]

            C_pose_root = C_img_path.split('/')[-2]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path.split('/')[-1].split('-')[0]
            C_identity = C_img_path.split('/')[-1].split('-')[1]
            C_back = C_img_path.split('/')[-1].split('-')[2]

            # B has same identity as C
            B_root = self.root.replace('train_img_c00_10class', 'vae_identity_new')
            B_category = C_category
            B_identity = C_identity
            B_img_root = os.path.join(B_root, B_category, B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)
            if not C_pose in B_img_name and not C_back in B_img_name:
                B_img_path = os.path.join(B_img_root, B_img_name)
            else:
                # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 1000
                continue  # break the BREAK_ALL

            # A has same pose as C
            A_img_root = os.path.join(self.root, C_pose_root)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            if not C_identity in A_img_name and not C_back in A_img_name:
                A_img_path = os.path.join(A_img_root, A_img_name)
            else:
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('train_img_c00_10class', 'vae_back_new')
            D_back = C_back
            # D has same back
            D_img_root_back = os.path.join(D_root, D_back)
            # D must have different identity and diff pose with C
            '''cate '''
            for roots, dirs, files in os.walk(D_img_root_back):
                cates = dirs
                break
            cates.remove(C_category)
            if len(cates) <= 0:  # no other category to choose
                # print('The D image can not have different cate with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL
            selected_D_cate = random.choice(cates)
            D_img_root_cate = os.path.join(D_img_root_back, selected_D_cate)
            '''pose '''
            for roots, dirs, files in os.walk(D_img_root_cate):
                poses = dirs
                break
            poses.remove(C_pose_root)



            if len(poses) <= 0:  # no other category to choose
                # print('The D image can not have different pose with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 2 if index < self.C_size - 20 else  index - 200
                continue  # break the BREAK_ALL
            selected_D_pose = random.choice(poses)
            D_img_root = os.path.join(D_img_root_cate, selected_D_pose)
            D_files = os.listdir(D_img_root)
            D_image_index = random.randint(0, len(D_files) - 1)
            D_img_path = os.path.join(D_img_root, D_files[D_image_index])

        return A_img_path, B_img_path, C_img_path, D_img_path
    def findtest(self, index):
        '''
        A: id provider
        B: pose provider
        D: background provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'id.jpg')
        B_img_path = os.path.join(group_path, 'pose.jpg')
        D_img_path = os.path.join(group_path, 'background.jpg')
        return A_img_path, B_img_path, D_img_path
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path = self.findABD(index)

            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            C_img = Image.open(C_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                C = self.transform(C_img)
                D = self.transform(D_img)

            return {'A': A, 'B': B, 'C': C, 'D': D}
        else: # test
            A_img_path, B_img_path, D_img_path = self.findtest(index)
            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                D = self.transform(D_img)

            return {'A': A, 'B': B, 'D': D}


    def __len__(self):
        return self.C_size

class Fonts_imgfolder(Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''
    def __init__(self, root, transform=None, train=True):
        super(Fonts_imgfolder, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        # self.paths = make_dataset(self.root)
        if self.train:
            self.C_size = 52 # too much we fix it as the number of letters
            '''refer'''
            # color 10
            self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
                      'cyan': (0, 255, 255),
                      'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
                      'silver': (192, 192, 192)}
            self.Colors = list(self.Colors.keys())
            # size 3
            self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
            self.Sizes = list(self.Sizes.keys())
            # style nearly over 100
            for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
                cates = dirs
                break
            self.All_fonts = cates
            print(len(self.All_fonts))
            print(self.All_fonts, len(self.All_fonts))
            # letter 52
            self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))]
        else: # test mode
            self.C_size, self.paths = group_path(self.root) # size of center image C

    def findN(self, index):
        # random choose a C image
        C_letter  = self.Letters[index]
        C_size = random.choice(self.Sizes)
        C_font_color = random.choice(self.Colors)
        resume_colors = self.Colors.copy()
        resume_colors.remove(C_font_color)
        C_back_color = random.choice(resume_colors)
        C_font = random.choice(self.All_fonts)
        C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)
        ''' exclusive the C attribute avoid same with C'''
        temp_Letters = self.Letters.copy()# avoid same size with C
        temp_Letters.remove(C_letter)
        temp_Size = self.Sizes.copy()# avoid same size with C
        temp_Size.remove(C_size)
        temp_font_color = self.Colors.copy()# avoid same font_color with C
        temp_font_color.remove(C_font_color)
        temp_back_colors = self.Colors.copy()  # avoid same back_color with C and avoid same color with font
        temp_back_colors.remove(C_back_color)
        temp_font = self.All_fonts.copy()  # avoid same font with C
        temp_font.remove(C_font)

        # A has same content
        '''SAME content'''
        A_letter = C_letter
        A_size = random.choice(temp_Size)
        A_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if A_font_color in resume_colors:
            resume_colors.remove(A_font_color)
        A_back_color = random.choice(resume_colors)
        A_font = random.choice(temp_font)
        A_img_name = A_letter + '_' + A_size + '_' + A_font_color + '_' + A_back_color + '_' + A_font + ".png"
        A_img_path = os.path.join(self.root, A_letter, A_size, A_font_color, A_back_color, A_font, A_img_name)

        # B has same size
        B_letter = random.choice(temp_Letters)
        '''SAME size'''
        B_size = C_size
        B_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if B_font_color in resume_colors:
            resume_colors.remove(B_font_color)
        B_back_color = random.choice(resume_colors)
        B_font = random.choice(temp_font)
        B_img_name = B_letter + '_' + B_size + '_' + B_font_color + '_' + B_back_color + '_' + B_font + ".png"
        B_img_path = os.path.join(self.root, B_letter, B_size, B_font_color, B_back_color, B_font, B_img_name)

        # D has same font_color
        D_letter = random.choice(temp_Letters)
        D_size = random.choice(temp_Size)
        '''SAME font_color'''
        D_font_color = C_font_color
        resume_colors = temp_back_colors.copy()
        if D_font_color in resume_colors:
            resume_colors.remove(D_font_color)
        D_back_color = random.choice(resume_colors)
        D_font = random.choice(temp_font)
        D_img_name = D_letter + '_' + D_size + '_' + D_font_color + '_' + D_back_color + '_' + D_font + ".png"
        D_img_path = os.path.join(self.root, D_letter, D_size, D_font_color, D_back_color, D_font, D_img_name)

        # E has same back_color
        E_letter = random.choice(temp_Letters)
        E_size = random.choice(temp_Size)
        resume_colors = temp_font_color.copy()
        resume_colors.remove(C_back_color)
        E_font_color = random.choice(resume_colors)
        '''SAME back_color'''
        E_back_color = C_back_color
        E_font = random.choice(temp_font)
        E_img_name = E_letter + '_' + E_size + '_' + E_font_color + '_' + E_back_color + '_' + E_font + ".png"
        E_img_path = os.path.join(self.root, E_letter, E_size, E_font_color, E_back_color, E_font, E_img_name)

        # F has same font
        F_letter = random.choice(temp_Letters)
        F_size = random.choice(temp_Size)
        F_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if F_font_color in resume_colors:
            resume_colors.remove(F_font_color)
        F_back_color = random.choice(resume_colors)
        '''SAME font'''
        F_font = C_font
        F_img_name = F_letter + '_' + F_size + '_' + F_font_color + '_' + F_back_color + '_' + F_font + ".png"
        F_img_path = os.path.join(self.root, F_letter, F_size, F_font_color, F_back_color, F_font, F_img_name)

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path
    def findtest(self, index):
        '''
                    refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style
                    A2B3D4E5F1_combine_2N
        A: size provider
        B: font_color provider
        D: back_color provider
        E: font provider
        F: letter provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'size.png')
        B_img_path = os.path.join(group_path, 'font_color.png')
        D_img_path = os.path.join(group_path, 'back_color.png')
        E_img_path = os.path.join(group_path, 'font.png')
        F_img_path = os.path.join(group_path, 'letter.png')

        return A_img_path, B_img_path, D_img_path, E_img_path, F_img_path
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path = self.findN(index)


            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            C_img = Image.open(C_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                C = self.transform(C_img)
                D = self.transform(D_img)
                E = self.transform(E_img)
                F = self.transform(F_img)

            return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F}
        else: # test
            A_img_path, B_img_path, D_img_path, E_img_path, F_img_path = self.findtest(index)

            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                D = self.transform(D_img)
                E = self.transform(E_img)
                F = self.transform(F_img)

            return {'A': A, 'B': B, 'D': D, 'E': E, 'F': F}

    def __len__(self):
        return self.C_size

class rafd_imgfolder(Dataset):
    def __init__(self, root, transform=None, train=True):
        super(rafd_imgfolder, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        print(root)
        if self.train:
            self.paths = make_dataset(self.root)
            self.C_size = len(self.paths) # size of center image C
        else: # test mode
            self.C_size, self.paths = group_path(self.root) # size of center image C
            print(self.C_size, self.paths, self.root)



    def findABD(self, index):
        FOUNDED = False
        while not FOUNDED:

            C_img_path = self.paths[index % self.C_size]
            # print(C_img_path.split('/'))
            #C_img_path = '/home2/RaFD/sep/data/27_090_neutral.jpg'
            #C_img_path ='/home2/RaFD/sep/data/45_045_surprised.jpg'

            files = os.listdir(self.root)
            E_img_name = random.choice(files)

            E_img_path = os.path.join(self.root, E_img_name)
            # print(E_img_path)
            '''
            local
            '''

            C_pose = C_img_path.split('/')[-1].split('_')[1]
            C_identity = C_img_path.split('/')[-1].split('_')[0]
            C_expression = C_img_path.split('/')[-1].split('_')[2].split('.')[0]

            B_root = self.root.replace('data', 'img_id')
            # B has same identity
            B_identity = C_identity

            B_img_root = os.path.join(B_root,  B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)

            b_founded = 0
            cnt = 0
            while not b_founded:
                cnt += 1
                if not C_pose == B_img_name.split('_')[1] and not C_expression == B_img_name.split('_')[2].split('.')[0]:
                    B_img_path = os.path.join(B_img_root, B_img_name)
                    b_founded =1
                else:
                    # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                    B_img_name = random.choice(B_files)
                    if cnt >= 100:
                        break
            if b_founded == 0:
                index += 1
                continue
            # A has same pose
            A_img_root = os.path.join(self.root.replace('data', 'img_pz'), C_pose)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            a_founded = 0
            cnt = 0
            while not a_founded:
                cnt += 1
                if C_identity != A_img_name.split('_')[0] and C_expression != A_img_name.split('_')[2].split('.')[0]:
                    A_img_path = os.path.join(A_img_root, A_img_name)
                    a_founded = 1
                else:
                    # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                    #     C_img_path))
                    A_img_name = random.choice(A_files)
                    if cnt >= 100:
                        break
            if a_founded == 0:
                index += 1
                continue
            # print(A_img_path)

            # D has same expression
            # back-cate-pose
            D_root = self.root.replace('data', 'img_ep')
            D_expression = C_expression
            # D has same back
            D_img_root = os.path.join(D_root, D_expression)
            D_files = os.listdir(D_img_root)
            D_img_name = random.choice(D_files)
            d_founded = 0
            cnt = 0
            while not d_founded:
                cnt += 1
                # print(C_img_path,'finding d')
                if not C_identity == D_img_name.split('_')[0] and not C_pose == D_img_name.split('_')[1]:
                    D_img_path = os.path.join(D_img_root, D_img_name)
                    d_founded = 1
                else:
                    # print('D must have different identity and diff pose with C)
                    D_img_name = random.choice(D_files)
                    if cnt >= 100:
                        break
            if d_founded == 0:
                index += 1
                continue
            # D must have different identity and diff pose with C
            '''cate '''
            # print(D_img_path)
            # print('---------------------------')
            FOUNDED = 1
            # check D
            # if D_identity in excluded_id:
            #     continue

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path

    def findtest(self, index):
        '''
        A: id provider
        B: pose provider
        D: expression provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'identity.png')
        B_img_path = os.path.join(group_path, 'pose.png')
        D_img_path = os.path.join(group_path, 'expression.png')
        return A_img_path, B_img_path, D_img_path

    def __getitem__(self, index):
        if self.train:
            '''there is a big while loop for choose category and training'''
            A_img_path, B_img_path, C_img_path, D_img_path, E_img_path = self.findABD(index)
            # A_img_path, B_img_path, C_img_path, D_img_path, E_img_path = self.find_test(index)

            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            C_img = Image.open(C_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                C = self.transform(C_img)
                D = self.transform(D_img)
                E = self.transform(E_img)

            return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}
        else: # test
            A_img_path, B_img_path, D_img_path = self.findtest(index)
            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                D = self.transform(D_img)

            return {'A': A, 'B': B, 'D': D}


    def __len__(self):
        return self.C_size


class dsprites_imgfolder(Dataset):
    '''
    shape: square, ellipse, heart / scale: 6 values linearly spaced in [0.5, 1]/ Orientation: 40 values in [0, 2 pi] /
    Position X: 32 values in [0, 1] / Position Y: 32 values in [0, 1]
    0: shape 1: scale 2:Orientation 3:X 4: Y
    E.g. square / 0.8 / 0 / 0.5 / 0.5
    C random sample
    AC same content; BC same size; DC same Orientation; EC same Position X; FC Position Y
    '''
    def __init__(self, root, transform=None, train=True):
        super(dsprites_imgfolder, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        # self.paths = make_dataset(self.root)
        if self.train:
            self.C_size = 737 # will not be used in training
            '''refer'''
            # Shape 3
            self.Shape = [str(n) for n in range(0, 3)]
            # Shape 3
            self.Scale = [str(n) for n in range(0, 6)]
            # Shape 3
            self.Orientation = [str(n) for n in range(0, 40)]
            # Shape 3
            self.X = [str(n) for n in range(0, 32)]
            # Shape 3
            self.Y = [str(n) for n in range(0, 32)]
        else: # test mode
            self.C_size, self.paths = group_path(self.root) # size of center image C


    def findN(self, index):
        # random choose a C image
        C_shape  = self.Shape[random.randint(0, 2)]
        C_scale = random.choice(self.Scale)
        C_orientation = random.choice(self.Orientation)
        C_X = random.choice(self.X)
        C_Y = random.choice(self.Y)
        C_img_name = C_shape + '_' + C_scale + '_' + C_orientation + '_' + C_X + '_' + C_Y + ".png"
        C_img_path = os.path.join(self.root, C_shape, C_scale, C_orientation, C_X, C_Y, C_img_name)
        ''' exclusive the C attribute avoid same with C'''
        temp_shape = self.Shape.copy()# avoid same size with C
        temp_shape.remove(C_shape)
        temp_scale = self.Scale.copy()# avoid same size with C
        temp_scale.remove(C_scale)
        temp_orientation = self.Orientation.copy()# avoid same font_color with C
        temp_orientation.remove(C_orientation)
        temp_X = self.X.copy()  # avoid same back_color with C and avoid same color with font
        temp_X.remove(C_X)
        temp_Y = self.Y.copy()  # avoid same font with C
        temp_Y.remove(C_Y)
        '''
        0: shape 1: scale 2:orientation 3:X 4: Y
        '''
        # A has same shape
        '''SAME shape '''

        A_shape = C_shape
        A_scale = random.choice(temp_scale)
        A_orientation = random.choice(temp_orientation)
        A_X = random.choice(temp_X)
        A_Y = random.choice(temp_Y)
        A_img_name = A_shape + '_' + A_scale + '_' + A_orientation + '_' + A_X + '_' + A_Y + ".png"
        A_img_path = os.path.join(self.root, A_shape, A_scale, A_orientation, A_X, A_Y, A_img_name)

        # B has same scale
        B_shape = random.choice(temp_shape)
        '''SAME scale'''
        B_scale = C_scale
        B_orientation = random.choice(temp_orientation)
        B_X = random.choice(temp_X)
        B_Y = random.choice(temp_Y)
        B_img_name = B_shape + '_' + B_scale + '_' + B_orientation + '_' + B_X + '_' + B_Y + ".png"
        B_img_path = os.path.join(self.root, B_shape, B_scale, B_orientation, B_X, B_Y, B_img_name)

        # D has same orientation
        D_shape = random.choice(temp_shape)
        D_scale = random.choice(temp_scale)
        '''SAME orientation'''
        D_orientation = C_orientation
        D_X = random.choice(temp_X)
        D_Y = random.choice(temp_Y)
        D_img_name = D_shape + '_' + D_scale + '_' + D_orientation + '_' + D_X + '_' + D_Y + ".png"
        D_img_path = os.path.join(self.root, D_shape, D_scale, D_orientation, D_X, D_Y, D_img_name)

        # E has same X
        E_shape = random.choice(temp_shape)
        E_scale = random.choice(temp_scale)
        E_orientation = random.choice(temp_orientation)
        '''SAME X'''
        E_X = C_X
        E_Y = random.choice(temp_Y)
        E_img_name = E_shape + '_' + E_scale + '_' + E_orientation + '_' + E_X + '_' + E_Y + ".png"
        E_img_path = os.path.join(self.root, E_shape, E_scale, E_orientation, E_X, E_Y, E_img_name)

        # F has same Y
        F_shape = random.choice(temp_shape)
        F_scale = random.choice(temp_scale)
        F_orientation = random.choice(temp_orientation)
        F_X = random.choice(temp_X)
        '''SAME Y'''
        F_Y = C_Y
        F_img_name = F_shape + '_' + F_scale + '_' + F_orientation + '_' + F_X + '_' + F_Y + ".png"
        F_img_path = os.path.join(self.root, F_shape, F_scale, F_orientation, F_X, F_Y, F_img_name)

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path
    def findtest(self, index):
        '''
                    refer 1: shape, 2: scale, 3: Orientation, 4 Position X, 5 Position Y
                    A2B3D4E5F1_combine_2N
        A: scale provider
        B: Orientation provider
        D: X provider
        E: Y provider
        F: shape provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'scale.png')
        B_img_path = os.path.join(group_path, 'orientation.png')
        D_img_path = os.path.join(group_path, 'X.png')
        E_img_path = os.path.join(group_path, 'Y.png')
        F_img_path = os.path.join(group_path, 'shape.png')

        return A_img_path, B_img_path, D_img_path, E_img_path, F_img_path
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path = self.findN(index)


            A_img = Image.open(A_img_path).convert('L')
            B_img = Image.open(B_img_path).convert('L')
            C_img = Image.open(C_img_path).convert('L')
            D_img = Image.open(D_img_path).convert('L')
            E_img = Image.open(E_img_path).convert('L')
            F_img = Image.open(F_img_path).convert('L')

            A = torch.from_numpy(np.array(A_img) / 255).unsqueeze(0).float()
            B = torch.from_numpy(np.array(B_img) / 255).unsqueeze(0).float()
            C = torch.from_numpy(np.array(C_img) / 255).unsqueeze(0).float()
            D = torch.from_numpy(np.array(D_img) / 255).unsqueeze(0).float()
            E = torch.from_numpy(np.array(E_img) / 255).unsqueeze(0).float()
            F = torch.from_numpy(np.array(F_img) / 255).unsqueeze(0).float()

            return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F}
        else: # test
            A_img_path, B_img_path, D_img_path, E_img_path, F_img_path = self.findtest(index)

            A_img = Image.open(A_img_path).convert('L')
            B_img = Image.open(B_img_path).convert('L')
            D_img = Image.open(D_img_path).convert('L')
            E_img = Image.open(E_img_path).convert('L')
            F_img = Image.open(F_img_path).convert('L')

            A = torch.from_numpy(np.array(A_img) / 255).unsqueeze(0).float()
            B = torch.from_numpy(np.array(B_img) / 255).unsqueeze(0).float()
            D = torch.from_numpy(np.array(D_img) / 255).unsqueeze(0).float()
            E = torch.from_numpy(np.array(E_img) / 255).unsqueeze(0).float()
            F = torch.from_numpy(np.array(F_img) / 255).unsqueeze(0).float()

            return {'A': A, 'B': B, 'D': D, 'E': E, 'F': F}

    def __len__(self):
        return self.C_size


class ilab_cumstom_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_cumstom_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)

    def findABD(self, index):
        FOUNDED = False
        while not FOUNDED:
            FOUNDED = True
            C_img_path = self.paths[index % self.C_size]
            '''
            local
            '''
            # print(C_img_path, C_img_path.split('/'))
            C_pose_root = C_img_path.split('/')[-2]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path.split('/')[-1].split('-')[0]
            C_identity = C_img_path.split('/')[-1].split('-')[1]
            C_back = C_img_path.split('/')[-1].split('-')[2]


            B_root = self.root.replace('train_img_c00_10class', 'vae_identity_new')
            # B has same identity
            B_category = C_category
            B_identity = C_identity
            B_img_root = os.path.join(B_root, B_category, B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)
            if not C_pose in B_img_name and not C_back in B_img_name:
                B_img_path = os.path.join(B_img_root, B_img_name)
            else:
                # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 1000
                continue  # break the BREAK_ALL

            index = index + 1 if index < self.C_size - 2 else index - 1000
            C_img_path1 = self.paths[index % self.C_size]
            '''
            local
            '''
            C_pose_root = C_img_path1.split('/')[-2]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path1.split('/')[-1].split('-')[0]
            C_identity = C_img_path1.split('/')[-1].split('-')[1]
            C_back = C_img_path1.split('/')[-1].split('-')[2]
            # A has same pose
            A_img_root = os.path.join(self.root, C_pose_root)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            if not C_identity in A_img_name and not C_back in A_img_name:
                A_img_path = os.path.join(A_img_root, A_img_name)
            else:
                # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                #     C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL

            index = index + 1 if index < self.C_size - 2 else index - 1000
            C_img_path2 = self.paths[index % self.C_size]
            '''
            local
            '''
            C_pose_root = C_img_path2.split('/')[-2]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path2.split('/')[-1].split('-')[0]
            C_identity = C_img_path2.split('/')[-1].split('-')[1]
            C_back = C_img_path2.split('/')[-1].split('-')[2]
            # D has same back
            # back-cate-pose
            D_root = self.root.replace('train_img_c00_10class', 'vae_back_new')
            D_back = C_back
            # D has same back
            D_img_root_back = os.path.join(D_root, D_back)
            # D must have different identity and diff pose with C
            '''cate '''
            for roots, dirs, files in os.walk(D_img_root_back):
                cates = dirs
                break
            cates.remove(C_category)
            if len(cates) <= 0:  # no other category to choose
                # print('The D image can not have different cate with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL
            selected_D_cate = random.choice(cates)
            D_img_root_cate = os.path.join(D_img_root_back, selected_D_cate)
            '''pose '''
            for roots, dirs, files in os.walk(D_img_root_cate):
                poses = dirs
                break
            # try:
            #     poses.remove(C_pose_root)
            # except:
            #     print(poses, C_pose_root)
            poses.remove(C_pose_root)



            if len(poses) <= 0:  # no other category to choose
                # print('The D image can not have different pose with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 2 if index < self.C_size - 20 else  index - 200
                continue  # break the BREAK_ALL
            selected_D_pose = random.choice(poses)
            D_img_root = os.path.join(D_img_root_cate, selected_D_pose)
            D_files = os.listdir(D_img_root)
            D_image_index = random.randint(0, len(D_files) - 1)
            D_img_path = os.path.join(D_img_root, D_files[D_image_index])

        return A_img_path, B_img_path, C_img_path, D_img_path, C_img_path1, C_img_path2
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        A_img_path, B_img_path, C_img_path, D_img_path, C_img_path1, C_img_path2 = self.findABD(index)


        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')
        D_img = Image.open(D_img_path).convert('RGB')
        C_1 = Image.open(C_img_path1).convert('RGB')
        C_2 = Image.open(C_img_path2).convert('RGB')


        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)
            D = self.transform(D_img)
            C_1 = self.transform(C_1)
            C_2 = self.transform(C_2)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'C1': C_1, 'C2': C_2}

    def __len__(self):
        return self.C_size


def return_data(args):
    name = args.dataset
    batch_size = args.batch_size
    # crop_size = args.crop_size
    image_size = args.image_size
    train = args.train
    if train:
        num_workers = args.num_workers
    else:
        num_workers = 1 # test mode
    # Create dataset
    if name.lower() == 'ilab_20m':
        if train: # train mode
            root = args.dataset_path
        else: # test mode
            root = os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No ilab-20M dataset')
        transform = [] # train test use same transform
        # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
        # transform.append(T.CenterCrop(crop_size)) # Do not need crop
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = ilab_imgfolder(root, transform, train)
    elif name.lower() == 'fonts':
        if train: # train mode
            root = args.dataset_path
        else: # test mode
            root = os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No fonts dataset')
        transform = []
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = Fonts_imgfolder(root, transform, train)
    elif name.lower() == 'rafd':
        print('selected rafd dataset')
        if train: # train mode
            root = args.dataset_path
        else: # test mode
            root = os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No rafd dataset')
        transform = []
        transform.append(T.Resize((image_size, image_size)))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = rafd_imgfolder(root, transform, train)
        print('rafd dataset found')

    elif name.lower() == 'dsprites':
        if train: # train mode
            root = args.dataset_path
        else: # test mode
            root = os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No dsprites dataset')
        transform = []
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = dsprites_imgfolder(root, transform, train)
    elif name.lower() == 'ilab_20m_custom':
        if train: # train mode
            root = args.dataset_path
        else: # test mode
            root = args.os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No ilab-20M-custom dataset')
        transform = [] # train test use same transform
        # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
        # transform.append(T.CenterCrop(crop_size)) # Do not need crop
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = ilab_cumstom_imgfolder(root, transform, train)

    else:
        raise NotImplementedError

    # Create dataloader
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers)

    return data_loader
