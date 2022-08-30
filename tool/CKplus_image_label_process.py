import os 
import numpy as np 
from tqdm import tqdm 



"""
Process CK+ images and labels to txt file in folder list 

"""

label_path = "../data/CK+/FACS"    
image_path = "../data/CK+/img"
list_path_prefix = "../data/CK+/list"    

#train_set = ['SN001', 'SN003', 'SN004', 'SN007', 'SN009', 'SN010', 'SN013']
#test_set = [ 'SN025', 'SN027']
train_set = []

test_set = os.listdir(image_path)

#au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
train_img_list = []
train_label_list = np.array([])
if len(train_set) != 0:
    for subject in tqdm(train_set):
        expressions = os.listdir(f'{label_path}/{subject}')
        for expression in expressions:
            imgs = os.listdir(f'{image_path}/{subject}/{expression}')

            expression_array = np.full((len(imgs), len(au_idx)), -1)
            for idx, img in enumerate(imgs):
                train_img_list.append(f'{subject}/{expression}/{img}')
                aus = []
                with open(f'{label_path}/{subject}/{expression}/{img[:-4]}_facs.txt') as f:
                    lines = f.readlines()
                    for line in lines:
                        aus.append(line.split()[0])

                    aus = [float(x) for x in aus]
                    aus = [int(x) for x in aus]
                    for au in aus:
                        if au in au_idx:
                            expression_array[idx, au_idx.index(au)] = 1


        if train_label_list.size == 0:
            train_label_list = expression_array
        else:
            train_label_list = np.append(train_label_list, expression_array, axis=0)


    train_label_list = train_label_list.astype(np.int32)

    np.savetxt(f'{list_path_prefix}/CK+_train_label.txt', train_label_list, fmt='%d', delimiter=' ')

    train_img_list = np.array(train_img_list)
    np.savetxt(f'{list_path_prefix}/CK+_train_img_path.txt', train_img_list, fmt='%s', delimiter='\n')


###############################################

test_img_list = []
test_label_list = np.array([])

for subject in tqdm(test_set):
    expressions = os.listdir(f'{label_path}/{subject}')
    for expression in expressions:
        imgs = os.listdir(f'{image_path}/{subject}/{expression}')

        expression_array = np.full((len(imgs), len(au_idx)), -1)
        for idx, img in enumerate(imgs):
            test_img_list.append(f'{subject}/{expression}/{img}')
            aus = []
            facs = os.listdir(f'{label_path}/{subject}/{expression}')
            with open(f'{label_path}/{subject}/{expression}/{facs[0]}') as f:
                lines = f.readlines()
                for line in lines:
                    if line != "\n":
                        aus.append(line.split()[0])

                aus = [float(x) for x in aus]
                aus = [int(x) for x in aus]
                for au in aus:
                    if au in au_idx:
                        expression_array[idx, au_idx.index(au)] = 1

        if test_label_list.size == 0:
            test_label_list = expression_array
        else:
            test_label_list = np.append(test_label_list, expression_array, axis=0)

test_label_list = test_label_list.astype(np.int32)

np.savetxt(f'{list_path_prefix}/CK+_test_label.txt', test_label_list, fmt='%d', delimiter=' ')

test_img_list = np.array(test_img_list)
np.savetxt(f'{list_path_prefix}/CK+_test_img_path.txt', test_img_list, fmt='%s', delimiter='\n')


