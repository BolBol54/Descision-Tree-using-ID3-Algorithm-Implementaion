import numpy as np
import pandas as pd
import math


def readData(url, col_names):
    return pd.read_csv(url, encoding="ISO-8859-1", header=None, names=col_names)  
        
def Pos_Neg_Num_all(data):
    label_column = data.keys()[-1]
    values = data[label_column].value_counts()
    
    if(values.shape == (1,) and values.keys() == True):
        p = values[True]
        n = 0
    elif(values.shape == (1,) and values.keys() == False):
        p = 0
        n = values[False]    
    else:
        p = values[True]
        n = values[False]
    return p, n

def Pos_Neg_Num(data, attribute, category):
    d = data.loc[data[attribute] == category]
    label_column = data.keys()[-1]
    values = d[label_column].value_counts()
    
    if(values.shape == (1,) and values.keys() == True):
        p = values[True]
        n = 0
    elif(values.shape == (1,) and values.keys() == False):
        p = 0
        n = values[False]    
    else:
        p = values[True]
        n = values[False]
    return p, n

def entropy(p, n):
    t1 = p / (p+n)
    t2 = n / (p+n)
    
    if(t1 == 0 or t2 == 0 ):
        return 0.0
    else:
        return -( t1*math.log2(t1) + t2*math.log2(t2))


def AverageInformationEntropy(data,  attribute):
    I = 0.0
    attribute_categories = data[attribute].unique()

    p_all , n_all = Pos_Neg_Num_all(data)
    
    for cat in attribute_categories:
        p, n = Pos_Neg_Num(data, attribute, cat)        
        en = entropy(p, n)
        I += ( (p+n) / (p_all+n_all )) * en
    return I

def Gain(data, attribute):
    I = AverageInformationEntropy(data, attribute)
    p , n = Pos_Neg_Num_all(data)
    entropy_s = entropy(p, n)
    return entropy_s - I

def find_winner(data):
    IG = []
    for key in data.keys()[:-1]:
        IG.append(Gain(data, key))

    max_gain_index = np.argmax(IG)    
    return data.keys()[:-1][max_gain_index]

def buildTree(data, tree=None):

    node = find_winner(data)

    attributeValue = np.unique(data[node])

    if tree is None:
        tree = {}
        tree[node] = {}

    for value in attributeValue:
        subsetData = data[data[node] == value]

        clValue,counts = np.unique(subsetData[subsetData.keys()[-1]],return_counts=True)
        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subsetData)

    return tree

def getList(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
          
    return list

def predict(tree, ex):
    
    flag = None
    k = getList(tree)
    key = k[0]    
    while(flag != True or False):
        
        tree = tree[key]
        value = ex[key]
        # print(key, value)
        if tree[value] == True or tree[value] == False:
            flag = tree[value]
            break
        else:
            tree = tree[value]
            k = getList(tree)
            key = k[0]

    return flag

def main():
    col_names = ["Outlook","Temp","Humidity","Wind","PlayTennis"]
    label_column = "PlayTennis"
    data = readData("data.csv", col_names)
    data_keys = data.keys()[:-1]

    import pprint
    t = buildTree(data)
    pprint.pprint(t)


    ex = data.iloc[5, :-1]
    p = predict(t, ex)
    print(p)

if __name__ == "__main__": main()


