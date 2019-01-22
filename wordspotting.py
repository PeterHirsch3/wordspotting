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
    #Dokument laden
    pageNumber = 2700270
    document_image_filename = 'pages/%d.png'%(pageNumber)
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    #Groundtruth laden
    gt = open("GT/%d.gtp"%(pageNumber))
    #Codebook laden
    input_file = open('codebook/codebook.bin', 'r')
    codebook = np.fromfile(input_file, dtype='float32')
    codebook = np.reshape(codebook, (4096,128))
    words = []
    #Groundtruth lesen
    for line in gt:
        str = line.rstrip().split()
        word_img = im_arr[int(str[1]):int(str[3]),int(str[0]):int(str[2])]
        words.append(word(word_img,str[4],codebook))
    gt.close()
    print "Deskriptoren berechnet"
    bof = []
    #BagofFeatures der Wörter zusammentragen
    for word_ in words:
        bof.append(word_.getBof())
    bof = np.array(bof)
    
    dim = {1,5,10,15,30,50,100,200}
    for d in dim:
        #Dimensionsreduktion in Dimension d
        tsp = TopicSubSpace(d)
        tsp.estimate(bof)
        bof_n = tsp.transform(bof)
        print bof_n.shape
        #Distanz der BoF der Wörter zueinander berechnen und sortieren
        dists = distance.cdist(bof_n,bof_n,'euclidean')
        dists = argsort(dists,axis = 1)
        
        result_word = np.array([([words[i].getWord() for i in c])for c in dists])
        result_img = np.array([([words[i].getImg() for i in c])for c in dists])    
        
        res = np.zeros(result_word.shape)
        i = 0;
        for c in result_word:
            j = 0
            for d in c:
                if c[0] == d:
                    res[i,j]=1
                j +=1
            i +=1
            
        print result_word[21,:25]
        print res[21,:25]
        ev = Eval()
        map = ev.mean_avarage_precision(res.tolist())
        print map
    
class word (object):
    
    def __init__(self, word_img, word,codebook):
        self._word_img_ = word_img
        self._word_ = word
        self._bof_ = None
        
        #Deskriptoren berechnen
        step_size = 5
        cell_size = 5        
        frames, desc = vlfeat.vl_dsift(word_img/255, step=step_size, size=cell_size)
        desc = np.array(desc, dtype=np.float)
        
        #Deskriptoren quantisieren mithilfe vom Codebook
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
   
if __name__ == '__main__':
    wordspotting()