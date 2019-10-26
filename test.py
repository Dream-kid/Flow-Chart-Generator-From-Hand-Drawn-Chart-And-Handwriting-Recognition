
import json
import time
import sys
from docx.shared import Inches
from docx import Document
def split1(ok):
    cnt=0 
    fo = open('words_dictionary.json','r')
    data = json.load(fo)
    with open(ok,'r') as f:
        for line in f:
            word = line.split()
            for temp in word:
                searchKey=temp.lower()
                if temp.isdigit():
                    cnt=cnt+1
                elif searchKey in data.keys():
                    cnt=cnt+1
                    print(temp)
                else:
                    print('-->',temp)   
    print('-------------------------')                
    return cnt         

     
    
def dell():    
   cnt1=split1('output/output1.txt')
   cnt2=split1('output/output2.txt')
   cnt3=split1('output/output3.txt')
   cnt4=split1('output/output4.txt')
   print(cnt1,cnt2,cnt3,cnt4)

dell()   
    