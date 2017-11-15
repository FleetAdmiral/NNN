
# coding: utf-8

# In[1]:


import io
def get_unigrams(file_name):
    unigrams = {}
    with io.open(file_name, encoding='utf8', errors='ignore') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                token = token.lower()
                try:
                    unigrams[token]
                except:
                    unigrams[token] = 0
                unigrams[token] += 1
                
    return unigrams

def index_unigrams(unigrams):
    new_unigrams = {}
    reverse_unigrams = {}
    for index, unigram in enumerate(unigrams):
        new_unigrams[unigram] = index
        reverse_unigrams[index] = unigram
    return new_unigrams, reverse_unigrams
            


# In[2]:


file_name = "sample_corpus.txt"
unigrams = get_unigrams(file_name)
iunigrams,runigrams = index_unigrams(unigrams)
unigrams = sorted(unigrams.items(), key = lambda x: x[1], reverse = True )
from pprint import pprint
#pprint.pprint(iunigrams) # Figure out non-stop words
dimensions = [x[0] for x in unigrams[100:3100]]
idimensions = {x: index for index, x in enumerate(dimensions)}
#pprint(dimensions)



# In[3]:


import numpy
cmatrix = numpy.memmap("lsa.cmatrix", dtype='float32', mode='w+', shape=(len(unigrams),len(dimensions)))
print(cmatrix.shape)


# In[ ]:


def populate_cmatrix(file_name, cmatrix, iunigrams, dimensions, window = 5):
     e = 0
     s = 0
     with open(file_name, encoding='utf-8', errors='ignore') as f:
        for index, line in enumerate(f):             
            tokens = line.strip().split()
            for indexj, token in enumerate(tokens):
                token = token.lower()
                lcontext = tokens[indexj - window:indexj]
                rcontext = tokens[indexj + 1:index + window]
                context = [tok.lower() for tok in lcontext + rcontext]
                
                try:
                    unigram_index = iunigrams[token]                    
                    for d in context:
                        
                        if d in dimensions:
                            j = dimensions[d]
                            cmatrix[unigram_index][j] += 1 
                            s += 1
                except:
                    e += 1
            
            
     print(e,s)
                
                


# In[ ]:


from time import time
s = time()
populate_cmatrix(file_name, cmatrix, iunigrams, idimensions)
e = time()
print(e -s)


# In[ ]:


w1 = 'eat'
w2 = 'drink'
w3 = 'print'
id1 = iunigrams[w1]
id2 = iunigrams[w2]
id3 = iunigrams[w3]
print(id1, id2, id3)
v1 = cmatrix[id1]
v2 = cmatrix[id2]
v3 = cmatrix[id3]

print(v1, v2, v3)

from scipy.spatial.distance import *
print(euclidean(v1, v2))
print(cosine(v1,v2))
print(cosine(v1,v3))


# In[ ]:


from sklearn.decomposition import TruncatedSVD
s = time()
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(cmatrix)
twod_cmatrix = svd.transform(cmatrix)
e = time()
print(e - s )


# In[ ]:


v1_2d, v2_2d = twod_cmatrix[id1], twod_cmatrix[id2]
id3 = iunigrams[w3]
v3_2d = twod_cmatrix[id3]
print(v1_2d, v2_2d, v3_2d)
print(cosine(v1_2d, v2_2d), cosine(v1_2d, v3_2d))


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
v1_2d = v1_2d / numpy.linalg.norm(v1_2d)
v2_2d = v2_2d / numpy.linalg.norm(v2_2d)
v3_2d = v3_2d / numpy.linalg.norm(v3_2d)
print ([v1_2d, v2_2d,v3_2d])
colors = ['r','b','g']
fig, axs = plt.subplots(1,1)
for i, x in enumerate([v1_2d, v2_2d,v3_2d]):
    a = plt.plot([0,x[0]],[0,x[1]],colors[i]+'-')
plt.show()

