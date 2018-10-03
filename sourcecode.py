

import pandas as pd
import math
train=pd.read_csv(r'C:\Users\uttej\Desktop\traindata.csv')

print(train['data'])

def createdict(y):
    x=y.data.str.split(" ")
    print(x.size)
    bag={}
    for each in range(x.size):
        for i in x[each]:
            if i in bag:
                bag[i]=bag[i]+1
            else:
                bag[i]=1
    
    
    print("Total vocabulary is "+str(len(bag)))
    print(train['class'])
    yes=0
    no=0
    for each in train['class']:
        if each == 1:
            yes=yes+1
        else:
            no=no+1
    return bag,yes,no,len(bag)
bag,yes,no,V = createdict(train)
print(bag)
print(yes)
print(no)
p0=[]
p1=[]
p0p={}
p1p={}
for word,clear in train.iterrows(): 
    if clear['class']==0:
        p0.append(clear['data'])
    else:
        p1.append(clear['data'])
for i in range(len(p0)):
    p0[i]=p0[i].split(" ")
    for j in range(len(p0[i])):
        if p0[i][j] in p0p:
            p0p[p0[i][j]]=p0p[p0[i][j]]+1
        else:
            p0p[p0[i][j]]=1

#print(p0)
for i in range(len(p1)):
    p1[i]=p1[i].split(" ")
    for j in range(len(p1[i])):
        if p1[i][j] in p1p:
            p1p[p1[i][j]]=p1p[p1[i][j]]+1
        else:
            p1p[p1[i][j]]=1
print(sum(p1p.values()))
######################################################################################################################################
#Testing
new=[]
newpred=[]
test = pd.read_csv(r'C:\Users\uttej\Desktop\testdata.txt', sep= "\n", header=None)
print(test)
for i in range(test.size):
    new.append(test.iloc[i][0])
print(new)
for i in range(len(new)):
    new[i]=new[i].split(" ")
print(new)
for u in range(len(new)):
    print(u)
    c0=math.log2(no/(yes+no))
    c1=math.log2(yes/(yes+no))
    for each in new[u]:
        try:
            c0=c0+math.log2((p0p[each]+1)/(V+sum(p0p.values())))
        except KeyError:
            c0=c0+math.log2(1/(V+sum(p0p.values())))
        try:
            c1=c1+math.log2((p1p[each]+1)/(V+sum(p1p.values())))
        except KeyError:
            c1=c1+math.log2(1/(V+sum(p1p.values())))
    print(c0)
    print(c1)
    if c0>c1:
        newpred.append(0)
    else:
        newpred.append(1)
print(newpred)
###CALULATING Accuracy
test_class=pd.read_csv(r'C:\Users\uttej\Desktop\testlabels.txt', sep= "\n", header=None)
pred=[]
for i in range(test_class.size):
    pred.append(test_class.iloc[i][0])
print(pred)
if len(pred)==len(newpred):
    acc=0
    for i in range(len(pred)):
        if pred[i]==newpred[i]:
            acc=acc+1
    print("*****Naive Bayes Multinomial Text Classifier***** \nTotal number of test instances are "+str(len(pred))+"\nCorrectly classified instances are "+str(acc)+"\nWrongly classified are "+str(len(pred)-acc)+"\nAccuracy is "+str(round((acc*100)/len(pred),2))+"%")
else:
    print("Please check for the correctness of the test data provided")