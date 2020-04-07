# Txt2Space

Load, explore and run basic operation on continuous (semantic) spaces saved in the txt format

Some supported functions and test: 
- cosine similarity
- knn extraction
- [Relpron](https://www.aclweb.org/anthology/J16-4004.pdf)
- [Mitchell & Lapata](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01106.x) 2010 test 
- [SimLex-999](https://fh295.github.io/simlex.html)
- space visualization (through PCA or [t-sne](https://lvdmaaten.github.io/tsne/))

## Basic instructions 

### File Format
Vectors have to be stored in the word - vector format, e.g.

```bash
en_word -0.07512683  0.0956306   0.12752604 -0.21758722  0.04896387 -0.3884378 ...
```

Be sure there is no space between line start and word nor commas between numbers.

### Usage
```python
from txt2space import Space

txt_space = 'path/to/space.txt'
space = Space()
space.txt2space(txt_space, en_remove=True)
```
Alternatively, if the first line of the file contains space informations (#tokens, #dimensions)

```python
space = Space()
space.txt2space(txt_space, dim_in_file=True)
space.wordsim_evaluations()
space.ml_eval()
```
```
Simlex,    coverage:994/999,   spearmanr:0.209, p:0.000
MEN (sim), coverage:2913/3000, spearmanr:0.724, p:0.000
WS353 sim, coverage:202/203,   spearmanr:0.772, p:0.000
WS353 rel, coverage:251/251,   spearmanr:0.652, p:0.000
relpron,   coverage:74665/323761, MAP value:0.107
testing adjectivenouns, coverage:1854/1944, spearmanr:0.464, p:0.000
testing verbobjects,    coverage:1944/1944, spearmanr:0.392, p:0.000
testing compoundnouns,  coverage:1944/1944, spearmanr:0.496, p:0.000
testing all,            coverage:5742/5832, spearmanr:0.449, p:0.000
```
```python
space.extract_knn('car', n_nbrs=10)
```
```
[('cars', 0.3988395306118516),
 ('earnhardt', 0.3751464141080185),
 ('racing', 0.3689658790679401),
 ('truck', 0.3547593171383764),
 ('driver', 0.34099113613900334),
 ('bike', 0.33193482489064463),
 ('vehicle', 0.3315263994468244),
 ('motorcycle', 0.32879053580788115),
 ('engine', 0.32823398561412326),
 ('automobile', 0.32311784173446095)]
```
```python
sapce.plot_space(method='tsne', word_count=1000, pick_random=True, size=(50, 50))
```
![space](https://github.com/lorenzoscottb/txt2space/blob/master/tests/semsp_test.png)

## Requirments
 - numpy >= 1.15.4
 - nltk >= 3.3
 - sklearn >= 0.20.0
