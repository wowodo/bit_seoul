import numpy as np
dataset = np.array(range(1, 11))
size = 5

def split_x(seq, size):
     aaa=[] #는 리스트
     for i in range(len(seq) - size + 1 ):
         subset = seq[i : (i+size)]
         aaa.append([item for item in subset]) # subset 만 넣어줘도 된다
     print(type(aaa))
     return np.array(aaa)

datasets = split_x(dataset, size)
print("===============")
print(datasets) 
