import numpy as np
import math

def sim(v1,v2):
    eud=np.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2+(v1[2]-v2[2])**2)
    mad=abs(v1[0]-v2[0])+abs(v1[1]-v2[1])+abs(v1[2]-v2[2])
    d1=np.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])
    d2=np.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])
    d1d2=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    coss=d1d2/(d1*d2)
    print(eud,mad,coss)

def gain(a,b):
    infod=-(a/(a+b))*math.log(a/(a+b),2)-(b/(a+b))*math.log(b/(a+b),2)
    # gini=1-a**2-b**2
    return infod


# a=np.array([1.4,1.6,9])
# b=np.array(([1.51,1.7,10],[2,1.9,8],[1.6,1.8,7],[1.2,1.5,11],[1.5,1.0,12]))
# for i in b:
#     print(i)
#     sim(a,i)

def meanvar(arr):
    sum=0
    for i in arr:
        sum+=i
    mean=(sum/len(arr))
    sum=0
    for i in arr:
        sum+=(i-mean)**2
    popvar=(1/len(arr))*sum
    samvar=(1/(len(arr)-1))*sum
    normal=[]
    for i in arr:
        inorm=(i-np.min(arr))/(np.max(arr)-np.min(arr))*(1-0)+0
        normal.append(inorm)
    standdev=math.sqrt(popvar)
    znorm=(100-mean)/standdev
    print('mean=',mean,'\n''popluation variance=',popvar,'\n''sample variance=',samvar,'\n''max-min normalization:',normal,'\n''the value of 100 in z-score normalization=',znorm)
    return mean,popvar,samvar,normal,znorm

# a=np.array([100, 400, 1000, 500, 2000])
a=gain(6,5)
c=gain(2,2) #age
d=gain(2,2) #salary
e=gain(1,1) #count
f=gain(1,1)
print('info gain of age=',a-(c*4/11))
print('info gain of salary=',a-(c*4/11))
print('info gain of count=',a-(e*2/11)-(f*2/11))