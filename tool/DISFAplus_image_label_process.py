import os 
import numpy as np 
from tqdm import tqdm 

"""
Process DISFA+ images and labels to txt file in folder list 
"""


label_path = "../data/DISFA+/Labels"
image_path = "../data/DISFA+/img"
list_path_prefix = "../data/DISFA+/list"

train_set = ['SN001', 'SN003', 'SN004', 'SN007', 'SN009', 'SN010', 'SN013']
test_set = [ 'SN025', 'SN027']


#au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
train_img_list = []
train_label_list = np.array([])

for subject in tqdm(train_set):
    expressions = os.listdir(f'{label_path}/{subject}')
    for expression in expressions:
        imgs = os.listdir(f'{image_path}/{subject}/{expression}')

        for img in imgs:
            train_img_list.append(f'{subject}/{expression}/{img}')
    
        expression_array = np.array([])
        for ai, au in enumerate(au_idx):
            AUlabel_path = f'{label_path}/{subject}/{expression}/AU{au}.txt'
            assert os.path.exists(AUlabel_path), f'No existing file {AUlabel_path}'

            img_name, au_label = np.array_split(np.loadtxt(AUlabel_path, dtype=str), [-1], axis=1)
            
            if expression_array.size == 0:
                expression_array = au_label
            else:
                expression_array = np.column_stack((expression_array, au_label))
        
        if train_label_list.size == 0:
            train_label_list = expression_array
        else:
            train_label_list = np.append(train_label_list, expression_array, axis=0)

train_label_list = train_label_list.astype(np.int32)
#train_label_list = np.where(train_label_list >= 2, 1, 0)
train_label_list = np.where(train_label_list >= 2, 1, -1)
np.savetxt(f'{list_path_prefix}/DISFA+_train_label.txt', train_label_list, fmt='%d', delimiter=' ')

train_img_list = np.array(train_img_list)
np.savetxt(f'{list_path_prefix}/DISFA+_train_img_path.txt', train_img_list, fmt='%s', delimiter='\n')


###############################################

test_img_list = []
test_label_list = np.array([])

for subject in tqdm(test_set):
    expressions = os.listdir(f'{label_path}/{subject}')
    for expression in expressions:
        imgs = os.listdir(f'{image_path}/{subject}/{expression}')

        for img in imgs:
            test_img_list.append(f'{subject}/{expression}/{img}')
    
        expression_array = np.array([])
        for ai, au in enumerate(au_idx):
            AUlabel_path = f'{label_path}/{subject}/{expression}/AU{au}.txt'
            assert os.path.exists(AUlabel_path), f'No existing file {AUlabel_path}'

            img_name, au_label = np.array_split(np.loadtxt(AUlabel_path, dtype=str), [-1], axis=1)
            
            if expression_array.size == 0:
                expression_array = au_label
            else:
                expression_array = np.column_stack((expression_array, au_label))
        
        if test_label_list.size == 0:
            test_label_list = expression_array
        else:
            test_label_list = np.append(test_label_list, expression_array, axis=0)

test_label_list = test_label_list.astype(np.int32)

#test_label_list = np.where(test_label_list >= 2, 1, 0)

test_label_list = np.where(test_label_list >= 2, 1, -1)

np.savetxt(f'{list_path_prefix}/DISFA+_test_label.txt', test_label_list, fmt='%d', delimiter=' ')

test_img_list = np.array(test_img_list)
np.savetxt(f'{list_path_prefix}/DISFA+_test_img_path.txt', test_img_list, fmt='%s', delimiter='\n')


