# PyTorch-Zero-Shot-Super-Resolution
An attempt at a PyTorch Implementation of "Zero-Shot" Super-Resolution using Deep Internal Learning by Shocher et al. CVPR 2018

I have tried to implement everything that has been given in the paper. Results are still lacking, but look promising. 
A lot of the preprocessing code has been taken directly from the original author's amazing repo, which is in tensorflow. 

Things left to do:
  1. Get results close to the paper
  2. Study effects of non-local, other attention based methods on this task.

Results on some images which have been resized to 2x size are given below.
The blurry image is the one obtained after bicubic interpolation while the sharper image is the one obtained by this implementation. 


![Alt text](images/Comic.gif?raw=true "Comic")


![Alt text](images/lincoln.gif?raw=true "Lincoln")
