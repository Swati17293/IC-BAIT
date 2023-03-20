from sklearn.model_selection import train_test_split
import pandas as pd


colnames = ['bias','headline','headline_pre','knowledge_p','knowledge_o']
data = pd.read_csv('Knowledge.csv', names=colnames, sep='\t')

features = data.drop('bias', axis=1)
categories = data.bias

A,B, a,b = train_test_split(features,categories,test_size=0.1, random_state = 0, stratify=categories, shuffle=True)

test = pd.concat([B, b], axis=1, join='inner')

test.to_csv('test.csv', sep='\t', index=False, header=False)



    
C,D, c,d = train_test_split(A,a,test_size=0.111, random_state = 0, stratify=a, shuffle=True)

train = pd.concat([C, c], axis=1, join='inner')

train.to_csv('train.csv', sep='\t', index=False, header=False)



valid = pd.concat([D, d], axis=1, join='inner')

valid.to_csv('valid.csv', sep='\t', index=False, header=False)