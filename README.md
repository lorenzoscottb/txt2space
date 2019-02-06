# Txt2Space

Load, explore and run basic operation on continuous (semantic) spaces from .txt format

Some supported functions and test: 
- cosine similarity
- knn extraction
- [Mitchell & Lapata](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01106.x) 2010 test 
- space visualization (through [t-sne](https://lvdmaaten.github.io/tsne/))

## Basic instructions 

Leave first line blank/for space information, as it will be discarded.
Vectors have to be stored in the word - vector format, e.g.

``
word -0.07512683  0.0956306   0.12752604 -0.21758722  0.04896387 -0.3884378 ...
``

Be sure there is no space between line start and word nor commas between numbers.

### Requirments
 - numpy >= 1.15.4
 - nltk >= 3.3
 - sklearn >= 0.20.0
