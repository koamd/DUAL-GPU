import glob
import cv2
import os
import argparse
import numpy as np
from DUAL import DUAL
import bm3d
import cupy as cp

dual = None

def setup_dual(iterations=20):
    global dual
    dual = DUAL(iterations=iterations, alpha=0.15, rho=1.1, gamma=0.5, limestrategy=2)

def run_dual(img_filename):

    global dual

    dual.load(img_filename)
    dual_result_np = dual.run()

    return dual_result_np

def bm3d_filter(img_noisy, sigma_psd):
    img_filt = bm3d.bm3d(img_noisy, sigma_psd, profile='np', stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return img_filt


def parse_args_and_config():
    
    parser = argparse.ArgumentParser(description='Batch Image enhancement for text quality analysis')
    parser.add_argument('--i', default='', type=str,
                        help='input path to image folder'
                        )
    parser.add_argument('--bm3d',action='store_true',
                        help='Use this argument to Use bm3d'
                        )
    
    args = parser.parse_args()

    return args

def main():

    args = parse_args_and_config()

    print('Cuda Devices available: ', cp.cuda.runtime.getDeviceCount())

    input_folder_img = args.i

    types = ('*.jpg', '*.png', '*.bmp') # the tuple of file types
    pic_files_original_img = []
    for ext in ('*.gif', '*.png', '*.jpg'):
        pic_files_original_img.extend(glob.glob(os.path.join(input_folder_img, ext)))

    setup_dual(iterations=30) #the more vibrant the colors, the more iteration you will need. For grayscale, we can set this to 10. 

    output_folder = 'output/dual'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sigma_psd = 20/255
        
    #average_loss = 0.0
    for index, filename in enumerate(pic_files_original_img):

        tail = os.path.split(filename)[-1]

        out_img = run_dual(filename)

        #perform clipping on output
        out_img = np.clip(out_img, 0, 1)

        #run bm3d filtering
        if args.bm3d : 

            cv2.imwrite(os.path.join(output_folder, 'dual_' +  tail), (out_img * 255).astype(np.uint8))

            print('[Info] Running BM3D on Dual output ')
            img_bm3d = bm3d_filter(out_img, sigma_psd)
            img_bm3d = np.clip(img_bm3d, 0, 1)
            img_bm3d_np = (img_bm3d*255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(output_folder, 'dual_bm3d_' + tail), img_bm3d_np)  
        else:
            out_img = (out_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, 'dual_' + tail), out_img)        

        print('[Info] Image {0} Completed'.format(filename))

if __name__ == "__main__":
    main()  



#python batchEnhancement.py --i /home/htx/Data/CBRNE/Contrast_enhancement_testing --bm3d
