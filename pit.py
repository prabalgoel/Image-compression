#from __future__ import print_function
exec('from __future__ import absolute_import, division, print_function')
#from __future__ import division
import numpy as np
import numpy.linalg as la
import numbers
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
from skimage import measure
#import time
#import itertools
from abc import abstractmethod
import pywt
from sklearn.preprocessing import normalize
from scipy.linalg import norm
from scipy import sum, average

try: 
    from itertools import accumulate 
except:
	# can also try numpy.cumsum
    import operator
    def accumulate(iterable, func=operator.add):
        'Return running totals'
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = func(total, element)
            yield total



class AbstractOperator(object):
    '''To make sure that the derived classes have the right functions'''
    @abstractmethod
    def apply(self, x):
        """Compute Ax"""
        pass
       
    @abstractmethod 
    def inv(self, x):
        """A^-1 x"""
        pass

class DCT(AbstractOperator):
    '''Discrete cosine transform'''
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, image):
        Timage = spfft.dct(spfft.dct(image, norm='ortho', axis=0, overwrite_x=True), norm='ortho', axis=1, overwrite_x=True)
        return Timage.reshape(-1)
    
    def inv(self, Timage):
        Timage = Timage.reshape(self.shape)
        return spfft.idct(spfft.idct(Timage, norm='ortho', axis=0), norm='ortho', axis=1)
    
class WT(AbstractOperator):
    '''wavelet transform: 
       call input: matrix
       inv input: vector of length fitting WT.shape'''
    def __init__(self, shape, wavelet = 'db6', level = 3, amplify = None):
        self.shape = shape
        self.wavelet = wavelet
        self.level = level
        self.cMat_shapes = [] 
        #build amplification vector of length 3*level
        if amplify is None:
            self.amplify = np.ones(3*self.level+1)
        else: 
            self.amplify = amplify
        if isinstance(amplify, numbers.Number):
            self.amplify = np.ones(3*self.level+1)
            self.amplify[0] = amplify       
    
    def __call__(self, image):
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=self.level)
        # format: [cAn, (cHn, cVn, cDn), ...,(cH1, cV1, cD1)] , n=level

        #to list of np.arrays
        #multiply with self.amplify[0] to have them more strongly weighted in compressions
        #tbd: implement others
        cMat_list = [coeffs[0]]
        for c in coeffs[1:]:
            cMat_list = cMat_list + list(c)
        #memorize all shapes for inv
        self.cMat_shapes = list(map(np.shape,cMat_list))
        
        #array vectorization
        vect = lambda array: np.array(array).reshape(-1)
        
        #store coeffcient matrices as vectors in list
        #cVec_list = map(vect,cMat_list)
        
        #apply amplification
        cVec_list = [vect(cMat_list[j])*self.amplify[j] for j in range(3*self.level+1)]
            
        return np.concatenate(cVec_list)
    
    def inv(self,wavelet_vector):
        '''Inverse WT
            cVec_list: vector containing all wavelet coefficients as vectrized in __call__'''
        
        #check if shapes of the coefficient matrices are known
        if self.cMat_shapes == []:
            print("Call WT first to obtain shapes of coefficient matrices")
            return None
        
        cVec_shapes = list(map(np.prod,self.cMat_shapes))
        
        split_indices = list(accumulate(cVec_shapes))
        
        cVec_list = np.split(wavelet_vector,split_indices)
        
        #reverse amplification
        cVec_list = [cVec_list[j]/self.amplify[j] for j in range(3*self.level+1)]

        #back to level format
        coeffs=[ np.reshape(cVec_list[0],self.cMat_shapes[0]) ]
        for j in range(self.level):
            triple = cVec_list[3*j+1:3*(j+1)+1]
            triple = [np.reshape( triple[i], self.cMat_shapes[1 +3*j +i] ) 
                     for i in range(3) ]
            coeffs = coeffs + [tuple(triple)]

        return pywt.waverec2( coeffs, wavelet=self.wavelet )
    
    def rand(self):
        '''outpus a random wavelet in picture domain'''
        Tz = self.__call__(np.zeros(shape)) # to initialize self.cMat_shapes
       
        cVec_shapes = list(map(np.prod,self.cMat_shapes))
        split_indices = list(accumulate(cVec_shapes))
        cVec_list = np.split(Tz,split_indices)
        
        #back to level format
        coeffs=[ np.reshape(cVec_list[0],self.cMat_shapes[0]) ]
        for j in range(self.level):
            triple = cVec_list[3*j+1:3*(j+1)+1]
            triple = [np.reshape( triple[i], self.cMat_shapes[1 +3*j +i] ) 
                     for i in range(3)]
            coeffs = coeffs + [tuple(triple)]
        
        return pywt.waverec2( coeffs, wavelet=self.wavelet )
    
#end class(WT)

        
def pltPic(X, size = (9,12) ):
    plt.figure(figsize=size)
    plt.imshow(X,interpolation='nearest', cmap=plt.cm.gray)
    plt.show()

    
class softTO(object):
    def __init__(self,tau):
        self.tau = tau
    def __call__(self,x):
        return pywt.threshold(x, self.tau, mode='soft')
    

def getRandMask(N,m):
    return np.random.choice(N, m, replace=False)





def proj(T, thOp, mask, Xsub, X):
    Xm = np.zeros(T.shape)
    Xm.flat[mask] = X.flat[mask]
    
    #calc gradient of squared L2-norm
    grad = 2*(Xm-Xsub)
    norm_grad = la.norm(grad.flat)
    
    #gradient step, transform
    TXnew = T( X-grad )
    TXnew = thOp(TXnew)
    return ( T.inv(TXnew), norm_grad )


def FISTA(T, thOp, mask, Xsub, stepsize = .8, n_steps = 100, X0=None, Xorig = None):
    if X0 is None:
        X = Xsub
    else:
        X = X0
    
    norm0 = la.norm(Xsub,'fro')
    
    if isinstance(Xorig,np.ndarray):
        print("Relative error : {:3.3f}".format( la.norm(X-Xorig,'fro')/la.norm(Xorig,'fro') ), end = '\n')
    t0 = stepsize/2 
    Y = X0
    for j in range(1,n_steps):
        X1, norm_grad = proj(T, thOp, mask, Xsub, Y)
        X1 = proj2range(X1)
        t1 = (1+np.sqrt( 1+4*t0**2 ))/2
        Y = X1 + ((t0-1)/t1)*(X1-X0)
        t0=t1
        X0=X1
        if j % 5 == 0:
            if isinstance(Xorig,np.ndarray):
                rel_error = la.norm(X1-Xorig,'fro')/la.norm(Xorig,'fro')
                if rel_error>10: break
                print("  {:3.3f}".format( rel_error ), end = '')            
            #interrupt if diverging
            elif la.norm(X,'fro')> 10*norm0*np.sqrt( np.prod(T.shape)/len(mask) ):
                break    
    print(' ')
    return X1

def proj2range(X):
    X = pywt.threshold(X, 255, mode='less', substitute = 255)
    X = pywt.threshold(X, 0, mode='greater', substitute = 0)
    return X

def compare_images(img1, img2):
    img11 = normalize(img1)
    img22 = normalize(img2)
    diff = img11 - img22
    m_norm =sum(abs(diff))
    z_norm =norm(diff.ravel(), 0)
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    s = measure.compare_ssim(img1, img2)
    #rel_error = la.norm(img2-img1,'fro')/la.norm(img1,'fro')
    return (m_norm, z_norm,err,s)