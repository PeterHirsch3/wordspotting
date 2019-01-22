import numpy as np
class Eval():
    
    
    def precision(self, ret_list):
         
        n_rel_val = float(ret_list.count(1))
        n_val = float(len(ret_list))
        precision = n_rel_val/n_val
        
        
        return precision
       
    def avarage_precision(self, ret_list, n_rel_data):
        sum = 0.0
        
        for k in range(len(ret_list)):
            rel_k = ret_list[k]
            prec_k = self.precision(ret_list[0:k+1])
            sum += rel_k * prec_k      
        
        av_prec = sum / n_rel_data
        
        return av_prec
    
    def mean_avarage_precision(self, ret_lists):
    
        sum_ap = 0.0
        
        for ret_list in ret_lists:
            sum_ap += self.avarage_precision(ret_list,float(ret_list.count(1)))
            
        m_a_p = sum_ap / len(ret_list)   
        
        return m_a_p
    
    def recall(self, ret_list, data):
        
        n_rel_val = float(ret_list.count(1))
        n_rel_val_data = float(len(data))
        recall = n_rel_val/n_rel_val_data
        
        return recall
    
    
    
    def mean_recall(self, ret_lists, data):
        
        sum_rec = 0.0
        
        for ret_list in ret_lists:
            sum_rec += self.recall(ret_list, data)
            
        m_r = sum_rec / len(ret_lists)
        
        return m_r
    