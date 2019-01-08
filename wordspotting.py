
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import matplotlib
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from scipy.signal.signaltools import correlate2d
from numpy import dtype
def wordspotting():
    pageNumber = 2730273
    document_image_filename = 'pages/%d.png'%(pageNumber)
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    plt.show()
    print im_arr
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
    step_size = 15
    cell_size = 3
    frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
if __name__ == '__main__':
    wordspotting()