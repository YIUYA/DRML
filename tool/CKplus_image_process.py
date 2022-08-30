from matplotlib import image
import scipy.io, scipy.misc 
import numpy as np 
import os 
from PIL import Image
from tqdm import tqdm 
from matplotlib import image
from matplotlib import pyplot as plt


"""
Cropped CK+ images using landmarks, and save to folder img
"""


face_landmark_path = "../data/CK+/Landmarks"

face_img_path = "../data/CK+/Original_img"

save_img_path = "../data/CK+/img"



def calculate_cropped_img_pos(landmark, offset = 0):
    bottom = landmark[:, 1].max() + offset
    top = landmark[:, 1].min() - offset
    left = landmark[:, 0].min() - offset
    right = landmark[:, 0].max()  + offset
    
    return top, bottom, left, right


subjects = os.listdir(face_img_path)
for subject in tqdm(subjects):
    if not os.path.isdir(f'{save_img_path}/{subject}'):
        os.mkdir(f'{save_img_path}/{subject}')
    expressions = os.listdir(f'{face_img_path}/{subject}')
    
    for expression in expressions:

        if '.D' not in expression:

            if not os.path.isdir(f'{save_img_path}/{subject}/{expression}'):
                os.mkdir(f'{save_img_path}/{subject}/{expression}')
            images = os.listdir(f'{face_img_path}/{subject}/{expression}')
            img = images[-1]  #Always use last image 
            lms = []
            with open(f'{face_landmark_path}/{subject}/{expression}/{img[:-4]}_landmarks.txt', 'r') as f:
                for line in f.readlines():
                    lms.append(line.split())
                
                lms = np.array(lms, np.float32)
            
            top, bottom, left, right = calculate_cropped_img_pos(lms, 5)
            im = Image.open(f'{face_img_path}/{subject}/{expression}/{img}')
            im = im.crop((left, top, right, bottom)) #cropped only face
            im = im.resize((200, 200), Image.ANTIALIAS) #make the same input as DISFA+

            im.save(f'{save_img_path}/{subject}/{expression}/{img[:-4]}.jpg')

                
print("Finish")