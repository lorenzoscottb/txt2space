# Txt2Space

Load, explore and run basic operation on continuous (semantic) spaces from .txt format

Supported functions include: 
- (cosine) similarity
- knn extraction
- automatic vectors addition
- space visualization (through t-sne)

## Basic instruction 

Leave first line blank/for space information, as it will be discarded.
Vectors have to be stored in the word - vector format, e.g.

``
space -0.07512683  0.0956306   0.12752604 -0.21758722  0.04896387 -0.3884378 ...
``

Be sure there is no space between line start and word or commas between numbers.

### Requirments
 - numpy >= 1.15.4
 - nltk >= 3.3
 - sklearn >= 0.20.0
