import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # IGNORE:unused-import
from matplotlib.patches import FancyArrowPatch
import numpy as np

class TopicSubSpace(object):
    
    def __init__(self, topic_dim):
        self.__topic_dim = topic_dim
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None
    def estimate(self, train_data):
        """Statistische Schaetzung des Topic Raums
        
        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        """
        
        T, S_arr, D_ = np.linalg.svd(train_data.T, full_matrices=False)
        S = np.diag(S_arr)
        
        self.__T = T[:,:self.__topic_dim] 
        self.__S_inv = np.linalg.inv(S)[:self.__topic_dim,:self.__topic_dim]
    
    def transform(self, data):
        """Transformiert Daten in den Topic Raum.
        
        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x topic_dim).
        """
        #   D    =      X'    *   T    *    S^-1
        return np.dot( np.dot( data, 
                               self.__T ), 
                       self.__S_inv )
        
        