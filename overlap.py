
import numpy as np
import pandas as pd

def findoverlap(list1,list2):
    # dist={}
    output=[]
    # for i in list1:
    #     temp=i[0:2]
    #     temp=str(temp)
    #     dist[temp]=1
    # print(dist)
    # for i in list2:
    #     temp1=i[0:2]
    #     temp1=str(temp1)
    #     if dist.get(temp1)==1:
    #         output.append(i)
    # print(output)
    # return output
    for i in list1:
        temp=i[0:2]
        dic={}
        dic1={}
        num=0
        for j in list2:
            temp1=j[0:2]
            lista=[]
            dist=abs(temp[0]-temp1[0])+abs(temp[1]-temp1[1])
            lista.append(dist)
            dic[dist]=j
            dic1[dist]=num
            num=num+1
        min=np.min(lista)
        fit=dic[min]
        index=dic1[min]
        list2=np.delete(list2,index,axis=0)
        print(len(list2))
        lop=np.append(i,fit)
        print(lop)
        output.append(np.append(i,fit))

    print(len(output))
    return output
        
dataset = pd.read_csv('2010.csv')
dataset1=pd.read_csv('2008.csv')
X = dataset.iloc[:, [0,1,2]].values
X1=dataset1.iloc[:,[0,1,2]].values
output=findoverlap(X1,X)