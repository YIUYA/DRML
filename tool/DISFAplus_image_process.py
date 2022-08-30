import scipy.io, scipy.misc 
import numpy as np 
import os 
from PIL import Image
from tqdm import tqdm 


"""
Cropped DISFA+ images (from *.mat file)and save to folder img
"""


face_landmark_path = "../data/DISFA+/FaceLandmarks"

save_img_path = "../data/DISFA+/img"


def save_img(face_landmark_path, subject_id, expression, save_img_path):

    
    mat = scipy.io.loadmat(f'{face_landmark_path}/{subject_id}/{expression}')

    #print ('here is the keys in the dictionary %s' %(mat.keys()))
    num_frames = np.shape(mat['FaceImg_CropResize'])[1] #this shows the number of frames for give sessions (facial expression)
    for frame_no in range(num_frames):
        #img_id = img = mat['FaceImg_CropResize'][0,frame_no][0,0][0] #image filename(.jpg) for the given frame (e.g. <frame_no>.jpg)
        cropped_img = mat['FaceImg_CropResize'][0,frame_no][0,0][1] #numpy.array for given image(200x200x3: RGB data) cropped face
        #landmark_pnts = mat['FaceImg_CropResize'][0,frame_no][0,0][2] # fourty nine landmark points coordinates [i.e. (x,y) values]

        img = Image.fromarray(cropped_img)
 
        img.save(f'{save_img_path}/{frame_no:03d}.jpg')

        #print ('landmark 49-points(for {image} the 200x200 pixel cropped face):\n {pnts} '.format(image=img_id, pnts=landmark_pnts))
        #print ('check the saved image (cropped 200x200 at {img_path})'.format(img_path=save_img_path))
        #break #comment this line to creat list of ALL of the images

subjects = os.listdir(face_landmark_path)
for subject in tqdm(subjects):
    if not os.path.isdir(f'{save_img_path}/{subject}'):
        os.mkdir(f'{save_img_path}/{subject}')
    expressions = os.listdir(f'{face_landmark_path}/{subject}')
    for expression in expressions:
        if not os.path.isdir(f'{save_img_path}/{subject}/{expression[:-16]}'):
            os.mkdir(f'{save_img_path}/{subject}/{expression[:-16]}')
        img_path = f'{save_img_path}/{subject}/{expression[:-16]}'
        save_img(face_landmark_path, subject, expression, img_path)
                
print("Finish")