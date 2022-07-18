temp=[]
for i in range (0,97):
    y=2227*7981**i%9311
    temp.append(y)
    
    

for i in range (0,50):
    y=(7**96)**i%9311
    print(i,y)
    for l in range (0,97):
        te=temp[l]
        if y == te:
            print(l, "true",te)
            
    

