# -*- coding: utf-8 -*-
import numpy as np;
#from numpy import insert;
import cv2;
import scipy;
from scipy import misc;
from scipy.misc import imread; 

#imag=scipy.misc.imread('F:\program file\canopy\Home_Work\lena_gray.png',1);

import numpy as np;
#from numpy import insert;
import cv2;
import scipy;
from scipy import misc;
from scipy.misc import imread; 
import random;

imag_origin=cv2.imread('F:\program file\canopy\Project\carriage.jpg',0);
[row,col]=imag_origin.shape;
#im=np.zeros(shape=(row,col),dtype=np.uint8);
#noise2=np.zeros(shape=(row,col),dtype=np.uint8);
#cv2.randn(noise2,(0),(20));
#print noise2;
imag=np.zeros(shape=(row,col));


############
##Important Declaration: this salt pepper noise function is from open source library:http://www.scriptscoop.net;
def impulse_noise(image,prob):

    fina_imag = np.zeros(image.shape,np.uint8)
    thre = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand = random.random()
            if rand < prob:
                fina_imag[i][j] = 240
            elif rand > thre:
                fina_imag[i][j] = 255
            else:
                fina_imag[i][j] = image[i][j]
    return fina_imag;

##############



imag = impulse_noise(imag_origin,0.02);
cv2.imwrite('ca_image_noise2.png', imag);



[row,col]=imag.shape;
row2=int(row*0.5);
col2=int(col*0.5);
row3=int(row2*0.5);
col3=int(col2*0.5);
row4=int(row3*0.5);
col4=int(col3*0.5);
row5=int(row4*0.5);
col5=int(col4*0.5);
row6=int(row5*0.5);
col6=int(col5*0.5);

imag_haar1=np.zeros(shape=(row,col));
haar_front1=np.zeros(shape=(row,row));
haar_back1=np.zeros(shape=(col,col));
haar_front2=np.zeros(shape=(row2,row2));
haar_back2=np.zeros(shape=(col2,col2));
haar_front3=np.zeros(shape=(row3,row3));
haar_back3=np.zeros(shape=(col3,col3));
haar_front4=np.zeros(shape=(row4,row4));
haar_back4=np.zeros(shape=(col4,col4));

grad_x1=np.zeros(shape=(row2,col2));
grad_y1=np.zeros(shape=(row2,col2));
grad_x2=np.zeros(shape=(row3,col3));
grad_y2=np.zeros(shape=(row3,col3));
grad_x3=np.zeros(shape=(row4,col4));
grad_y3=np.zeros(shape=(row4,col4));


grad_x1_rs=np.zeros(shape=(row,col));
grad_y1_rs=np.zeros(shape=(row,col));
grad_x2_rs=np.zeros(shape=(row2,col2));
grad_y2_rs=np.zeros(shape=(row2,col2));
grad_x3_rs=np.zeros(shape=(row3,col3));
grad_y3_rs=np.zeros(shape=(row3,col3));

def canny_thre(edge_image,row,col,thre1,thre2):
    
#    for i in range(0,row):
 #       for j in range(0,col):
  #          if edge_image[i][j]<thre2:
   #             edge_image[i][j]=0;
    
    
    mark=np.zeros(shape=(row,col));
    for i in range(0,row):
        for j in range(0,col):
            if edge_image.item((i,j))<thre1 and edge_image.item((i,j))>thre2:
                mark[i][j]=1;
            elif edge_image.item((i,j))>thre1:
                mark[i][j]=2;
            elif edge_image.item((i,j))<thre2:
                mark[i][j]=0;
    
    
    change_sign=0;
    while change_sign==1:
        for i in range(0,row):
            for j in range(0,col):
                if mark[i][j]==2:
                    if mark[i-1][j-1]==1:
                        mark[i-1][j-1]==2;
                        change_sign=1;
                    if mark[i+1][j-1]==1:
                        mark[i+1][j-1]==2;
                        change_sign=1;
                    if mark[i-1][j+1]==1:
                        mark[i-1][j+1]==2;
                        change_sign=1;
                    if mark[i+1][j+1]==1:
                        mark[i+1][j+1]==2;
                        change_sign=1;
                        
    
    new_edge=np.zeros(shape=(row,col));           
    for i in range(0,row):
        for j in range(0,col):
            if mark[i][j]==2:
                new_edge[i][j]=0;
            else:
                new_edge[i][j]=255;
    
    
    return new_edge;









def canny_thre_grad(edge_image,grad_x,grad_y,row,col,thre1,thre2):
    
#    for i in range(0,row):
 #       for j in range(0,col):
  #          if edge_image[i][j]<thre2:
   #             edge_image[i][j]=0;
    
    
    mark=np.zeros(shape=(row,col));
    for i in range(0,row):
        for j in range(0,col):
            if edge_image.item((i,j))<thre1 and edge_image.item((i,j))>thre2:
                mark[i][j]=1;
            elif edge_image.item((i,j))>thre1:
                mark[i][j]=2;
            elif edge_image.item((i,j))<thre2:
                mark[i][j]=0;
    
    
    change_sign=0;
    while change_sign==1:
        for i in range(0,row):
            for j in range(0,col):
                if mark[i][j]==2:
                    tang=grad_y[i][j]/grad_x[i][j];
                    if mark[i+1][j]==1 and tang<0.414 and tang>-0.414:
                        mark[i+1][j]==2;
                        change_sign=1;
                    if mark[i-1][j]==1 and tang<0.414 and tang>-0.414:
                        mark[i-1][j]==2;
                        change_sign=1;
                    if mark[i-1][j+1]==1 and tang<-0.414 and tang>-2.414:
                        mark[i-1][j+1]==2;
                        change_sign=1;
                    if mark[i+1][j-1]==1 and tang<-0.414 and tang>-2.414:
                        mark[i+1][j-1]==2;
                        change_sign=1;
                    if (mark[i+1][j]==1 and tang>2.414) or (mark[i+1][j]==1 and tang<-2.414):
                        mark[i+1][j]==2;
                        change_sign=1;
                    if (mark[i-1][j]==1 and tang>2.414) or (mark[i-1][j]==1 and tang<-2.414):
                        mark[i-1][j]==2;
                        change_sign=1;
                    if mark[i-1][j-1]==1 and tang>0.414 and tang<2.414:
                        mark[i-1][j-1]==2;
                        change_sign=1;
                    if mark[i+1][j+1]==1 and tang>0.414 and tang<2.414:
                        mark[i+1][j+1]==2;
                        change_sign=1;
  
    
                                                     
    new_edge=np.zeros(shape=(row,col));           
    for i in range(0,row):
        for j in range(0,col):
            if mark[i][j]==2:
                new_edge[i][j]=0;
            else:
                new_edge[i][j]=255;
    
    
    return new_edge;




def canny_thre_grad_test(edge_image,grad_x,grad_y,row,col,thre1,thre2):
    
#    for i in range(0,row):
 #       for j in range(0,col):
  #          if edge_image[i][j]<thre2:
   #             edge_image[i][j]=0;
    
    
    mark=np.zeros(shape=(row,col));
    for i in range(0,row):
        for j in range(0,col):
            if edge_image.item((i,j))<thre1 and edge_image.item((i,j))>thre2:
                mark[i][j]=1;
            elif edge_image.item((i,j))>thre1:
                mark[i][j]=2;
            elif edge_image.item((i,j))<thre2:
                mark[i][j]=0;
    
    
    change_sign=0;
    while change_sign==1:
        for i in range(0,row):
            for j in range(0,col):
                if mark[i][j]==2:
                    tang=grad_y[i][j]/grad_x[i][j];
                    if mark[i][j+1]==1 and tang<0.414 and tang>-0.414:
                        mark[i][j+1]==2;
                        change_sign=1;
                    if mark[i][j-1]==1 and tang<0.414 and tang>-0.414:
                        mark[i][j-1]==2;
                        change_sign=1;
                    if mark[i-1][j+1]==1 and tang<-0.414 and tang>-2.414:
                        mark[i-1][j+1]==2;
                        change_sign=1;
                    if mark[i+1][j-1]==1 and tang<-0.414 and tang>-2.414:
                        mark[i+1][j-1]==2;
                        change_sign=1;
                    if (mark[i+1][j]==1 and tang>2.414) or (mark[i+1][j]==1 and tang<-2.414):
                        mark[i+1][j]==2;
                        change_sign=1;
                    if (mark[i-1][j]==1 and tang>2.414) or (mark[i-1][j]==1 and tang<-2.414):
                        mark[i-1][j]==2;
                        change_sign=1;
                    if mark[i-1][j-1]==1 and tang>0.414 and tang<2.414:
                        mark[i-1][j-1]==2;
                        change_sign=1;
                    if mark[i+1][j+1]==1 and tang>0.414 and tang<2.414:
                        mark[i+1][j+1]==2;
                        change_sign=1;
  
                   
                                                     
    new_edge=np.zeros(shape=(row,col));           
    for i in range(0,row):
        for j in range(0,col):
            if mark[i][j]==2:
                new_edge[i][j]=0;
            else:
                new_edge[i][j]=255;
    
    
    return new_edge;








    
def norm_thre(edge_image,row,col,thre):
    
     new_edge=np.zeros(shape=(row,col));
     for i in range(0,row):
        for j in range(0,col):
            if edge_image.item((i,j))<thre:
                new_edge[i][j]=255;
            else:
                new_edge[i][j]=0;      #255-edge_image.item((i,j));
    
     return new_edge;
                
                    
                  
def norm_thre2(edge_image,row,col,thre):
    
     new_edge=np.zeros(shape=(row,col));
     for i in range(0,row):
        for j in range(0,col):
            if edge_image.item((i,j))<thre:
                new_edge[i][j]=0;
            else:
                new_edge[i][j]=edge_image.item((i,j));
    
     return new_edge;          
    


def non_maxi_sup(edge_imag,grad_x,grad_y,row,col):
    result=np.zeros(shape=(row,col));
    for i in range(1,row-1):
            for j in range(1,col-1):
                result[i][j]=edge_imag.item((i,j));
    for i in range(1,row-1):
            for j in range(1,col-1):
                #print edge_imag.item((i,j));
                if grad_x.item((i,j))!=0:
                    tang=grad_y.item((i,j))/grad_x.item((i,j));
                    if tang<0.414 and tang>-0.414:
                        if Max(edge_imag,i,j,1)!=1:
                            result[i][j]=0;
                    if tang<-0.414 and tang>-2.414:
                        if Max(edge_imag,i,j,2)!=1:
                            result[i][j]=0;
                    if (tang>2.414) or (tang<-2.414):
                        if Max(edge_imag,i,j,3)!=1:
                            result[i][j]=0;
                    if (tang>0.414) and (tang<2.414):
                        if Max(edge_imag,i,j,4)!=1:
                            result[i][j]=0;
                
                        
    return result;

    
def Max(edge_imag,i,j,k):
    sign=0;
    if k==1:
        if edge_imag.item((i,j))>edge_imag.item((i+1,j)) and edge_imag.item((i,j))>edge_imag.item((i-1,j)):
            sign=1;
    if k==2:
        if edge_imag.item((i,j))>edge_imag.item((i+1,j+1)) and edge_imag.item((i,j))>edge_imag.item((i-1,j-1)):
            sign=1;
    if k==3:
        if edge_imag.item((i,j))>edge_imag.item((i,j+1)) and edge_imag.item((i,j))>edge_imag.item((i,j-1)):
            sign=1;
    if k==4:
        if edge_imag.item((i,j))>edge_imag.item((i-1,j+1)) and edge_imag.item((i,j))>edge_imag.item((i+1,j-1)):
            sign=1;
            
    return sign;








##########################################
#first scale

for i in range(0,row):
    for j in range(0,row):
        if i<row2 and (j==2*i or j==2*i+1):
            haar_front1[i][j]=0.5;
        elif i>=row2 and j==2*(i-row2): 
            haar_front1[i][j]=-0.5;
        elif i>=row2 and j==2*(i-row2)+1:
            haar_front1[i][j]=0.5;


for i in range(0,col):
    for j in range(0,col):
        if i<col2 and (j==2*i or j==2*i+1):
            haar_back1[i][j]=0.5;
        elif i>=col2 and j==2*(i-col2): 
            haar_back1[i][j]=-0.5;
        elif i>=col2 and j==2*(i-col2)+1:
            haar_back1[i][j]=0.5;

haar_back1=haar_back1.transpose();

haar_front1=np.mat(haar_front1);
haar_back1=np.mat(haar_back1);

imag_haar1=haar_front1*imag*haar_back1;






imag_haar_abs1=np.zeros(shape=(row,col));
for i in range(0,row):
    for j in range(0,col):
        Num=abs(imag_haar1.item((i,j)));
        imag_haar_abs1[i][j]=Num;



for i in range(0,row):
    for j in range(0,col):
        if i>=row2 and j<col2:
            grad_x1[i-row2][j]=imag_haar1.item((i,j));
        if i<row2 and j>=col2:
            grad_y1[i][j-col2]=imag_haar1.item((i,j));



grad_x1_rs=cv2.resize(grad_x1,(row,col));
grad_y1_rs=cv2.resize(grad_y1,(row,col));






cv2.imwrite('ca_haar1.png',imag_haar_abs1);



#imag_haar_edge1=np.zeros(shape=(row,col));
#for i in range(0,row):
 #   for j in range(0,col):
  #      if i<row2 and j<col2:
   #          imag_haar_edge1[i][j]=0;
    #    else:  
     #       imag_haar_edge1[i][j]=imag_haar_abs1.item((i,j));



#imag_haar_edge_compute1=np.linalg.inv(haar_front1)*imag_haar_edge1*np.linalg.inv(haar_back1);
#cv2.imwrite('ca_haar_edge1.png',imag_haar_edge_compute1);


##########################################
#second scale

imag2=np.zeros(shape=(row2,col2));
for i in range(0,row2):
    for j in range(0,col2):
        imag2[i][j]=imag_haar1.item((i,j));


for i in range(0,row2):
    for j in range(0,row2):
        if i<row3 and (j==2*i or j==2*i+1):
            haar_front2[i][j]=0.5;
        elif i>=row3 and j==2*(i-row3): 
            haar_front2[i][j]=-0.5;
        elif i>=row3 and j==2*(i-row3)+1:
            haar_front2[i][j]=0.5;
            


for i in range(0,col2):
    for j in range(0,col2):
        if i<col3 and (j==2*i or j==2*i+1):
            haar_back2[i][j]=0.5;
        elif i>=col3 and j==2*(i-col3): 
            haar_back2[i][j]=-0.5;
        elif i>=col3 and j==2*(i-col3)+1:
            haar_back2[i][j]=0.5;


haar_back2=haar_back2.transpose();

haar_front2=np.mat(haar_front2);
haar_back2=np.mat(haar_back2);

imag_haar2=haar_front2*imag2*haar_back2;
imag_haar_abs2=np.zeros(shape=(row2,col2));

for i in range(0,row2):
    for j in range(0,col2):
        Num=abs(imag_haar2.item((i,j)));
        imag_haar_abs2[i][j]=Num;

cv2.imwrite('ca_haar2.png',imag_haar_abs2);



for i in range(0,row2):
    for j in range(0,col2):
        if i>=row3 and j<col3:
            grad_x2[i-row3][j]=imag_haar2.item((i,j));
        if i<row3 and j>=col3:
            grad_y2[i][j-col3]=imag_haar2.item((i,j));




grad_x2_rs=cv2.resize(grad_x2,(row2,col2));
grad_y2_rs=cv2.resize(grad_y2,(row2,col2));





imag_haar_resize2=np.zeros(shape=(row,col));
imag_haar_resize2=cv2.resize(imag_haar_abs2,(row,col));


imag_haar_mult1=np.zeros(shape=(row,col));
for i in range(0,row):
    for j in range(0,col):
        Num=(imag_haar_abs1.item((i,j))*imag_haar_resize2.item((i,j)))**(0.5);
        imag_haar_mult1[i][j]=Num;

cv2.imwrite('ca_haar_mult1.png',imag_haar_mult1);

imag_mult_edge1=np.zeros(shape=(row,col));
for i in range(0,row):
    for j in range(0,col):
        if i<row2 and j<col2:
             imag_mult_edge1[i][j]=0;
        else:  
            imag_mult_edge1[i][j]=imag_haar_mult1.item((i,j));



cv2.imwrite('ca_mult_edge1.png',imag_mult_edge1);


imag_edge1_comb=np.linalg.inv(haar_front1)*imag_mult_edge1*np.linalg.inv(haar_back1);
cv2.imwrite('ca_edge1_comb.png',imag_edge1_comb);


imag_edge_thre_grad1=canny_thre_grad_test(imag_edge1_comb,grad_x1_rs,grad_y1_rs,row,col,20,5);

imag_edge_comb_thre1=canny_thre(imag_edge1_comb,row,col,20,5);

imag_norm_thre1=norm_thre2(imag_edge1_comb,row,col,24);




cv2.imwrite('ca_edge_comb_thre1.png',imag_edge_comb_thre1);

cv2.imwrite('ca_edge_thre_grad1.png',imag_edge_thre_grad1);

cv2.imwrite('ca_norm_thre1.png',imag_norm_thre1);



#cv2.imwrite('ca_edge_grad_sup.png',imag_edge_grad_sup);





#imag_haar_edge2=np.zeros(shape=(row2,col2));
#for i in range(0,row2):
 #   for j in range(0,col2):
  #      if i<row3 and j<col3:
   #          imag_haar_edge2[i][j]=0;
    #    else:  
     #       imag_haar_edge2[i][j]=imag_haar_abs2.item((i,j));



#imag_haar_edge_compute2=np.linalg.inv(haar_front2)*imag_haar_edge2*np.linalg.inv(haar_back2);
#cv2.imwrite('ca_haar_edge2.png',imag_haar_edge_compute2);


##########################################
#third scale

imag3=np.zeros(shape=(row3,col3));
for i in range(0,row3):
    for j in range(0,col3):
        imag3[i][j]=imag_haar2.item((i,j));


for i in range(0,row3):
    for j in range(0,row3):
        if i<row4 and (j==2*i or j==2*i+1):
            haar_front3[i][j]=0.5;
        elif i>=row4 and j==2*(i-row4): 
            haar_front3[i][j]=-0.5;
        elif i>=row4 and j==2*(i-row4)+1:
            haar_front3[i][j]=0.5;
            

for i in range(0,col3):
    for j in range(0,col3):
        if i<col4 and (j==2*i or j==2*i+1):
            haar_back3[i][j]=0.5;
        elif i>=col4 and j==2*(i-col4): 
            haar_back3[i][j]=-0.5;
        elif i>=col4 and j==2*(i-col4)+1:
            haar_back3[i][j]=0.5;


haar_back3=haar_back3.transpose();

haar_front3=np.mat(haar_front3);
haar_back3=np.mat(haar_back3);

imag_haar3=haar_front3*imag3*haar_back3;
imag_haar_abs3=np.zeros(shape=(row3,col3));

for i in range(0,row3):
    for j in range(0,col3):
        Num=abs(imag_haar3.item((i,j)));
        imag_haar_abs3[i][j]=Num;



cv2.imwrite('ca_haar3.png',imag_haar_abs3);


for i in range(0,row3):
    for j in range(0,col3):
        if i>=row4 and j<col4:
            grad_x3[i-row4][j]=imag_haar3.item((i,j));
        if i<row4 and j>=col4:
            grad_y3[i][j-col4]=imag_haar3.item((i,j));


grad_x3_rs=cv2.resize(grad_x3,(row3,col3));
grad_y3_rs=cv2.resize(grad_y3,(row3,col3));


imag_haar_resize3=np.zeros(shape=(row2,col2));
imag_haar_resize3=cv2.resize(imag_haar_abs3,(row2,col2));



imag_haar_mult2=np.zeros(shape=(row2,col2));
for i in range(0,row2):
    for j in range(0,col2):
        Num=(imag_haar_abs2.item((i,j))*imag_haar_resize3.item((i,j)))**(0.5);
        imag_haar_mult2[i][j]=Num;

cv2.imwrite('ca_haar_mult2.png',imag_haar_mult2);




imag_mult_edge2=np.zeros(shape=(row2,col2));
for i in range(0,row2):
    for j in range(0,col2):
        if i<row3 and j<col3:
             imag_mult_edge2[i][j]=0;
        else:  
            imag_mult_edge2[i][j]=imag_haar_mult2.item((i,j));



cv2.imwrite('ca_mult_edge2.png',imag_mult_edge2);


imag_edge2_comb=np.linalg.inv(haar_front2)*imag_mult_edge2*np.linalg.inv(haar_back2);
cv2.imwrite('ca_edge2_comb.png',imag_edge2_comb);

imag_edge_thre_grad2=canny_thre_grad(imag_edge2_comb,grad_x2_rs,grad_y2_rs,row2,col2,20,5);

imag_edge_comb_thre2=canny_thre(imag_edge2_comb,row2,col2,20,5);

imag_norm_thre2=norm_thre2(imag_edge2_comb,row2,col2,24);

cv2.imwrite('ca_edge_comb_thre2.png',imag_edge_comb_thre2);

cv2.imwrite('ca_edge_thre_grad2.png',imag_edge_thre_grad2);

cv2.imwrite('ca_norm_thre2.png',imag_norm_thre2);




#imag_haar_edge3=np.zeros(shape=(row3,col3));
#for i in range(0,row3):
 #   for j in range(0,col3):
  #      if i<row4 and j<col4:
   #          imag_haar_edge3[i][j]=0;
    #    else:  
     #       imag_haar_edge3[i][j]=imag_haar_abs3.item((i,j));



#imag_haar_edge_compute3=np.linalg.inv(haar_front3)*imag_haar_edge3*np.linalg.inv(haar_back3);
#cv2.imwrite('ca_haar_edge3.png',imag_haar_edge_compute3);





##########################################
#fourth scale

imag4=np.zeros(shape=(row4,col4));
for i in range(0,row4):
    for j in range(0,col4):
        imag4[i][j]=imag_haar3.item((i,j));


for i in range(0,row4):
    for j in range(0,row4):
        if i<row5 and (j==2*i or j==2*i+1):
            haar_front4[i][j]=0.5;
        elif i>=row5 and j==2*(i-row5): 
            haar_front4[i][j]=-0.5;
        elif i>=row5 and j==2*(i-row5)+1:
            haar_front4[i][j]=0.5;
            


for i in range(0,col4):
    for j in range(0,col4):
        if i<col5 and (j==2*i or j==2*i+1):
            haar_back4[i][j]=0.5;
        elif i>=col5 and j==2*(i-col5): 
            haar_back4[i][j]=-0.5;
        elif i>=col5 and j==2*(i-col5)+1:
            haar_back4[i][j]=0.5;
            
            
haar_back4=haar_back4.transpose();            

haar_front4=np.mat(haar_front4);
haar_back4=np.mat(haar_back4);

imag_haar4=haar_front4*imag4*haar_back4;
imag_haar_abs4=np.zeros(shape=(row4,col4));

for i in range(0,row4):
    for j in range(0,col4):
        Num=abs(imag_haar4.item((i,j)));
        imag_haar_abs4[i][j]=Num

cv2.imwrite('ca_haar4.png',imag_haar_abs4);



imag_haar_resize4=np.zeros(shape=(row3,col3));
imag_haar_resize4=cv2.resize(imag_haar_abs4,(row3,col3));


imag_haar_mult3=np.zeros(shape=(row3,col3));
for i in range(0,row3):
    for j in range(0,col3):
        Num=(imag_haar_abs3.item((i,j))*imag_haar_resize4.item((i,j)))**(0.5);
        imag_haar_mult3[i][j]=Num;

cv2.imwrite('ca_haar_mult3.png',imag_haar_mult3);



imag_mult_edge3=np.zeros(shape=(row3,col3));
for i in range(0,row3):
    for j in range(0,col3):
        if i<row4 and j<col4:
             imag_mult_edge3[i][j]=0;
        else:  
            imag_mult_edge3[i][j]=imag_haar_mult3.item((i,j));



cv2.imwrite('ca_mult_edge3.png',imag_mult_edge3);





imag_edge3_comb=np.linalg.inv(haar_front3)*imag_mult_edge3*np.linalg.inv(haar_back3);
cv2.imwrite('ca_edge3_comb.png',imag_edge3_comb);

imag_edge_thre_grad3=canny_thre_grad(imag_edge3_comb,grad_x3_rs,grad_y3_rs,row3,col3,20,5);

imag_edge_comb_thre3=canny_thre(imag_edge3_comb,row3,col3,20,5);

imag_norm_thre3=norm_thre2(imag_edge3_comb,row3,col3,24);

cv2.imwrite('ca_edge_comb_thre3.png',imag_edge_comb_thre3);

cv2.imwrite('ca_edge_thre_grad3.png',imag_edge_thre_grad3);

cv2.imwrite('ca_norm_thre3.png',imag_norm_thre3);




#imag_haar_edge4=np.zeros(shape=(row4,col4));
#for i in range(0,row4):
 #   for j in range(0,col4):
  #      if i<row5 and j<col5:
   #          imag_haar_edge4[i][j]=0;
    #    else:  
     #       imag_haar_edge4[i][j]=imag_haar_abs4.item((i,j));



#imag_haar_edge_compute4=np.linalg.inv(haar_front4)*imag_haar_edge4*np.linalg.inv(haar_back4);
#cv2.imwrite('ca_haar_edge4.png',imag_haar_edge_compute4);


imag_norm_comb_resize2=np.zeros(shape=(row,col));
imag_norm_comb_resize3=np.zeros(shape=(row,col));

imag_norm_comb_resize2=cv2.resize(imag_edge2_comb,(row,col));
imag_norm_comb_resize3=cv2.resize(imag_edge3_comb,(row,col));

imag_norm_comb=np.zeros(shape=(row,col));
imag_norm_comb=imag_norm_thre1+imag_norm_comb_resize2+imag_norm_comb_resize3;



imag_norm_thre=norm_thre(imag_norm_comb,row,col,15);

cv2.imwrite('ca_norm_thre.png',imag_norm_thre);


imag_norm_thre=norm_thre(imag_norm_comb,row,col,15);

cv2.imwrite('ca_norm_comb.png',imag_norm_comb);




imag_canny_comb_resize2=np.zeros(shape=(row,col));
imag_canny_comb_resize3=np.zeros(shape=(row,col));

imag_canny_comb_resize2=cv2.resize(imag_edge2_comb,(row,col));
imag_canny_comb_resize3=cv2.resize(imag_edge3_comb,(row,col));

imag_canny_comb=np.zeros(shape=(row,col));
imag_canny_comb=imag_edge1_comb+imag_canny_comb_resize2+imag_canny_comb_resize3;

imag_canny_thre=canny_thre(imag_canny_comb,row,col,31,10);

cv2.imwrite('ca_canny_thre.png',imag_canny_thre);






imag_edge_comb_sup1=non_maxi_sup(imag_edge1_comb,grad_x1_rs,grad_y1_rs,row,col);
imag_edge_comb_sup2=non_maxi_sup(imag_edge2_comb,grad_x2_rs,grad_y2_rs,row2,col2);
imag_edge_comb_sup3=non_maxi_sup(imag_edge3_comb,grad_x3_rs,grad_y3_rs,row3,col3);



imag_canny_comb_sup_resize2=np.zeros(shape=(row,col));
imag_canny_comb_sup_resize3=np.zeros(shape=(row,col));

imag_canny_comb_sup_resize2=cv2.resize(imag_edge_comb_sup2,(row,col));
imag_canny_comb_sup_resize3=cv2.resize(imag_edge_comb_sup3,(row,col));



imag_canny2_thre_sup=np.zeros(shape=(row,col));


imag_canny2_comb_sup=imag_edge_comb_sup1+imag_canny_comb_sup_resize2+imag_canny_comb_sup_resize3;

imag_canny2_thre_sup=canny_thre(imag_canny2_comb_sup,row,col,31,10);

cv2.imwrite('ca_canny2_thre_sup.png',imag_canny2_thre_sup);










