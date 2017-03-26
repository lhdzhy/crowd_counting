# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:08:01 2017

@author: lhd
"""
'''
In general terms if you really care about getting the the exact same result as MATLAB, the easiest way to achieve this is often by looking directly at the source of the MATLAB function.

In this case, edit fspecial:

...
  case 'gaussian' % Gaussian filter

     siz   = (p2-1)/2;
     std   = p3;

     [x,y] = meshgrid(-siz(2):siz(2),-siz(1):siz(1));
     arg   = -(x.*x + y.*y)/(2*std*std);

     h     = exp(arg);
     h(h<eps*max(h(:))) = 0;

     sumh = sum(h(:));
     if sumh ~= 0,
       h  = h/sumh;
     end;
...
Pretty simple, eh? It's <10mins work to port this to Python:
This gives me the same answer as fspecial to within rounding error:
'''

import numpy as np

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
