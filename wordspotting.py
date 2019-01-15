
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import matplotlib
import vlfeat
import scipy.spatial.distance as distance
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from scipy.signal.signaltools import correlate2d
from numpy import dtype
from DokAn_wordspotting.LSI import TopicSubSpace
def wordspotting():
    pageNumber = 2700270
    document_image_filename = 'pages/%d.png'%(pageNumber)
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    plt.show()
    gt = open("GT/%d.gtp"%(pageNumber))
    word_img = []
    words = []
    for line in gt:
        str = line.rstrip().split()
        word_img.append(im_arr[int(str[1]):int(str[3]),int(str[0]):int(str[2])])
        words.append(str[4])
    gt.close()
    print words[23]
    plt.imshow(word_img[23],cmap = cm.get_cmap('Greys_r'))
    plt.show()
    step_size = 5
    cell_size = 5
    
    frames, desc = vlfeat.vl_dsift(word_img[23]/255, step=step_size, size=cell_size)
    desc = np.array(desc, dtype=np.float)
    print desc
    input_file = open('codebook/codebook.bin', 'r')
    codebook = np.fromfile(input_file, dtype='float32')
    codebook = np.reshape(codebook, (4096,128))
    print frames
    dist_mat = distance.cdist(desc, codebook,'euclidean')
    dist_mat_sort_ind = np.argsort(dist_mat, axis=1)
    global_ = dist_mat_sort_ind[:,0]
    left = global_[:len(global_)/2]
    right = global_[len(global_)/2:]
    print global_,left,right
    bof_g = np.bincount(global_)
    bof_l = np.bincount(left)
    bof_r = np.bincount(right)
    #concatinate
    bof = bof_g + bof_l + bof_r
if __name__ == '__main__':
    wordspotting()