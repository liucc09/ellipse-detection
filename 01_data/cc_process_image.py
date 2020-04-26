import imageio
import skimage
import numpy as np
from skimage.color import rgb2gray
from skimage.util.dtype import img_as_ubyte
from skimage.filters import scharr
from skimage.filters import rank
from skimage.morphology import disk
from skimage import exposure
from skimage.color import gray2rgb
from skimage.color import rgb2hsv

def get_all_images(path,names,suffix,itype='gray'):
    ims = []
    for n in names:
        fpath = f'{path}/{n}.{suffix}'
        im = imageio.imread(fpath)
        if itype=='gray':
            im = rgb2gray(im)
        elif itype=='rgb' and len(im.shape)==2:
            im = gray2rgb(im)
        elif itype=='hsv' and len(im.shape)==2:
            im = gray2rgb(im)
            im = rgb2hsv(im)
        elif itype=='hsv' and len(im.shape)==3:
            im = rgb2hsv(im)
            
        im = img_as_ubyte(im)
        
        ims.append(im)
        
    return ims

def preprocess_image(im,scale=False):
    if np.max(im)>1:
        im = im/255
    
    if scale:
        im_min = np.min(im)
        im_max = np.max(im)
        
        im = (im-im_min)/(im_max-im_min)
    
    return im

def preprocess_images(ims,scale=False):
    
    imos = []
    for im in ims:
        im = preprocess_image(im,scale)
        imos.append(im)
        
    return imos

def postprocess_image(im,thr=0.8):
    if np.max(im)>1:
        return np.where(im>thr*255,1,0)
    else:
        return np.where(im>thr,1,0)
    
def postprocess_images(ims,thr=0.8):
    
    imos = []
    for im in ims:
        
        im = postprocess_image(im,thr)
        imos.append(im)
        
    return imos