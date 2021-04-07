#encoding: utf-8
'''
re oganize a dsprites dataset for trainding
shape: square, ellipse, heart / scale: 6 values linearly spaced in [0.5, 1]/ Orientation: 40 values in [0, 2 pi] /
Position X: 32 values in [0, 1] / Position Y: 32 values in [0, 1]
E.g. square / 0.8 / 0 / 0.5 / 0.5
potential : position (x, y) bold, rotation

'''
import os
import numpy as np
from PIL import Image
import matplotlib
import scipy.misc
import random

import shutil
from tqdm import tqdm

# dsprite_dir = '/home2/dsprites_dataset'
dsprite_dir = '/lab/tmpig23b/u/andy/dsprites'
output_dsprite_dir = '/lab/tmpig23b/u/andy/dsprites/dsprites'

# read raw data
root = os.path.join(dsprite_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
data = np.load(root, encoding='bytes')
all_imgs = data['imgs']
latents_classes = data['latents_classes']


pbar = tqdm(total=len(all_imgs))

for index, array in enumerate(all_imgs):
    # create the path:
    shape = str(latents_classes[index][1])
    scale = str(latents_classes[index][2])
    orientation = str(latents_classes[index][3])
    X = str(latents_classes[index][4])
    Y = str(latents_classes[index][5])
    save_path = os.path.join(output_dsprite_dir, shape, scale, orientation, X, Y)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save image
    im = Image.fromarray(all_imgs[index] * 255)
    im_name = shape +'_'+ scale +'_'+ orientation +'_'+ X +'_'+ Y + '.png'
    im.save(os.path.join(save_path, im_name))

    # print
    pbar.update(1)
    pbar.write('[{}] '.format(index))
pbar.write("[Process Finished]")
pbar.close()






























'''reference'''
# color 10 (back ground and font)
Colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30), 'silver': (192,192,192)}
# size 3
Sizes = {'small': 80, 'medium' : 100, 'large': 120}
# style nearly over 100
All_fonts = pygame.font.get_fonts()
useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',  'mrykacstqurn', 'ori1uni','pothana2000','vemana2000',
                'navilu', 'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari']
useless_fontsets = ['kacst', 'lohit', 'sam']
# throw away the useless
for useless_font in useless_fonts:
    All_fonts.remove(useless_font)
temp = All_fonts.copy()
for useless_font in temp: # check every one
    for set in useless_fontsets:
        if set in useless_font:
            try:
                All_fonts.remove(useless_font)
            except:
                print(useless_font)
# letter 52
Letters = list(range(65, 91)) + list(range(97, 123))
img_size = 128


font_dir = '/home2/dsprites_dataset'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

pygame.init()
screen = pygame.display.set_mode((img_size, img_size)) # image size Fix(128 * 128)


for letter in Letters: # 1st round for letters
    for size in Sizes.keys():  # 2nd round for size
        for font_color in Colors.keys():  # 3rd round for font_color
            for back_color in Colors.keys():  # 4th round for back_color
                # if not back_color == font_color:''' should not be same '''
                for font in All_fonts:  # 5th round for fonts
                    if not font_color == back_color:
                        try:
                            # 1 set back_color
                            screen.fill(Colors[back_color]) # background color
                            # 2 set letter
                            selected_letter = chr(letter)
                            # 3,4 set font and size
                            selected_font = pygame.font.SysFont(font, Sizes[size]) # size and bold or not
                            # 5 set font_color
                            rtext = selected_font.render(selected_letter, True, Colors[font_color], Colors[back_color])

                            # screen.blit(rtext, (img_size/2, img_size/2))
                            # screen.blit(rtext, (img_size / 4, 0))
                            screen.blit(rtext, (10, 0)) # because
                            # E.g. A / 64/ red / blue / arial
                            img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' + font + ".png"
                            img_path = os.path.join(font_dir, selected_letter, size, font_color, back_color, font)
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            pygame.image.save(screen, os.path.join(img_path, img_name))
                        except:
                            print(letter, size, font_color, back_color, font)
                    else:
                        break








# screen.fill((255,255,255)) # background color
# start, end = (97, 255) # 汉字编码范围
# for codepoint in range(int(start), int(end)):
#     word = chr(codepoint)
#     font = pygame.font.SysFont("arial", 64) # size and bold or not
#     # font = pygame.font.Font("msyh.ttc", 64)
#     rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
#     # pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
#     screen.blit(rtext, (300, 300))
#     pygame.image.save(screen, os.path.join(chinese_dir, word + ".png"))
