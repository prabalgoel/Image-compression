from pit import *
import scipy.ndimage as spimg
from scipy.misc import imsave
import glob
import os
import imageio
import sys
sys.path.append("..")
pic_file = "./pics/3.jpg"
Xorig = imageio.imread(pic_file, as_gray=True)

x = pywt.wavedec2(Xorig, wavelet='db1', level=1)
Xorig = pywt.waverec2(x, wavelet='db1')

shape = Xorig.shape
n = np.prod(shape)

wlet = 'db12'
L = 3 
amp = np.linspace(1,.2,L)
#print("----",amp)
amp = np.kron(amp, np.ones(3) )
amp = np.insert(amp,0, 20 ) 
             
#print("----",amp)

print("Original image:")
pltPic(Xorig)

p = .05

m = int(p*n)
print("Total number of pixels: {}k".format(int(np.prod(Xorig.shape)/1000)))
print("Number of remaining pixels: {}k".format( int(m/1000) ))

mask = getRandMask(n, m)
Xsub = np.zeros(shape)
Xsub.flat[mask] = Xorig.flat[mask]

print("Masked image:")
pltPic(Xsub)

steps_wt = 14
steps_dct = 10 
N = 7 

th_wt  = np.append( np.linspace(25,4,N-1), 4 )
th_dct = np.append( np.linspace(15,3,N-1), 3 )

if 'Xrec' not in locals():
    Xrec = Xsub
    
Xrec = Xsub

dct = DCT(shape)
wt = WT(shape, wavelet=wlet,level=L, amplify=amp)

for j in range(N):
    thOp = softTO(th_dct[j])
    Xrec=FISTA(dct, thOp, mask, Xsub, stepsize = .75, n_steps=steps_dct, Xorig=Xorig, X0=Xrec)
    
    thOp = softTO(th_wt[j])
    Xrec=FISTA(wt, thOp, mask, Xsub, stepsize = .75, n_steps=steps_wt, Xorig=Xorig, X0=Xrec)

print("Reconstructed image:")
pltPic(Xrec)


Xsub2 = Xsub.copy()
Xsub2[Xsub2==0.0] = Xsub2.max()

imageio.imwrite("./pics/3_masked.jpg", Xsub2.astype(np.uint8))
imageio.imwrite("./pics/3_rec.jpg", Xrec.astype(np.uint8))


img1 = Xorig.astype(float)
img2 = imageio.imread("./pics/3_rec.jpg").astype(float)
n_m, n_0,err,s = compare_images(img1, img2)
print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)
print("MSE:",err)
print("SSIM:",s)
rel_error = la.norm(img2-img1,'fro')/la.norm(img1,'fro')
print("Relative compression error: {}",rel_error)
