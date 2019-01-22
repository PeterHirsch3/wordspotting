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
from numpy import dtype, argsort
from DokAn_wordspotting.LSI import TopicSubSpace
from eval import Eval
from features import RelativeTermFrequencies
def wordspotting():
    pageNumber = 2700270
    document_image_filename = 'pages/%d.png'%(pageNumber)
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    #plt.show()
    gt = open("GT/%d.gtp"%(pageNumber))
    input_file = open('codebook/codebook.bin', 'r')
    codebook = np.fromfile(input_file, dtype='float32')
    codebook = np.reshape(codebook, (4096,128))
    words = []
    for line in gt:
        str = line.rstrip().split()
        word_img = im_arr[int(str[1]):int(str[3]),int(str[0]):int(str[2])]
        words.append(word(word_img,str[4],codebook))
    gt.close()
    plt.imshow(words[23].getImg(),cmap=cm.get_cmap('Greys_r'))
    #plt.show()
    print "Deskriptoren berechnet"
    bof = []
    for word_ in words:
        bof.append(word_.getBof())
    bof = np.array(bof)
    rtf = RelativeTermFrequencies()
    bof = rtf.weighting(bof)
    #dim = {5}
    #for d in dim:
        #tsp = TopicSubSpace(d)
        #tsp.estimate(bof)
        #bof = tsp.transform(bof)
    dists = distance.cdist(bof,bof,'euclidean')
    dists = argsort(dists,axis = 1)
    result_word = np.array([([words[i].getWord() for i in c])for c in dists])
    result_img = np.array([([words[i].getImg() for i in c])for c in dists])
    print result_word[21,:20]
    res = np.zeros(result_word.shape)
    i = 0;
    for c in result_word:
        j = 0
        for d in c:
            if c[0] == d:
                res[i,j]=1
            j +=1
        i +=1
    print res[21,:20]
    ev = Eval()
    map = ev.mean_avarage_precision(res.tolist())
    print map
    
class word (object):
    
    def __init__(self, word_img, word,codebook):
        self._word_img_ = word_img
        self._word_ = word
        self._bof_ = None
        step_size = 5
        cell_size = 5
    
        frames, desc = vlfeat.vl_dsift(word_img/255, step=step_size, size=cell_size)
        desc = np.array(desc, dtype=np.float)
    
        dist_mat = distance.cdist(desc, codebook,'euclidean')
        dist_mat_sort_ind = np.argsort(dist_mat, axis=1)
        global_ = dist_mat_sort_ind[:,0]
        left = global_[:len(global_)/2]
        #mid = global_[len(global_)/3:2*len(global_)/3]
        right = global_[len(global_)/2:]
        bof_g = np.bincount(global_,minlength = 4095)
        bof_l = np.bincount(left,minlength = 4095)
        #bof_m = np.bincount(mid,minlength = 4095)
        bof_r = np.bincount(right,minlength = 4095)
        self._bof_ = np.concatenate((bof_g,bof_l,bof_r),axis = 0)
    def getWord(self):
        return self._word_
    def getBof(self):
        return self._bof_
    def getImg(self):
        return self._word_img_
    def setBof(self,bof):
        self._bof_ = bof

    
if __name__ == '__main__':
    wordspotting()