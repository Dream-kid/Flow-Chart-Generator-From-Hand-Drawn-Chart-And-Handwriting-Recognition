import json
import time
import sys
def func(word)
fo = open('words_dictionary.json','r')
data = json.load(fo)
searchKey = 'if'
startTime=time.time()
if searchKey in data.keys():
	print("Match Found!")
else:
	print("Not found")
#print("No.of words: ",len(data.keys()))
#print("Run Time: ",time.time()-startTime,' seconds')