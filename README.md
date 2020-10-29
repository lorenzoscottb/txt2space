# Txt2Space

This library allows you to load (static) vector spaces, saved in .txt format, and run multiple qualitative and quantitative investigations. Tests include word similarity/relatedness benchmarks, as well as visualisations analysis of the spaces.

Some supported functions and test: 
- cosine similarity
- knn extraction
- space visualization (through PCA or [t-sne](https://lvdmaaten.github.io/tsne/))

word-word tasks:
- [SimLex-999](https://fh295.github.io/simlex.html)
- [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN)
- [WordSim353](http://alfonseca.org/eng/research/wordsim353.html)

Composition based tasks
- [Relpron](https://www.aclweb.org/anthology/J16-4004.pdf)
- [Mitchell & Lapata](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01106.x) 2010 test 
- [Big BiRD](http://saifmohammad.com/WebPages/BiRD.html)

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
The en_remove argument, if True, will remove the prefix from words label during uploading. 

If the first line of the file contains space's informations (i.e. #tokens, #features), passsing the dim_in_file arguments as True will automatically update the information for upload. 

```python
space = Space()
space.txt2space(txt_space, dim_in_file=True)
space.run_tests(relpron_ds='dev', mix_gr_rel=True)
```
Tasks can be run invidually, or collectively, as shown below. For each mode, the retunr_results argumet will allow to store correlation variables (i.e correlation value, p value). Coverage indicates how many items from the origianl datasets have been tested. If even a single word from an item is missing, the all item will be ignored for the evaluaiton. 
```
Simlex,    coverage:726/999,   spearmanr:0.140, p:0.000
MEN (sim), coverage:1544/3000, spearmanr:0.628, p:0.000
WS353 sim, coverage:152/203,   spearmanr:0.615, p:0.000
WS353 rel, coverage:200/251,   spearmanr:0.564, p:0.000
relpron test,           coverage:19152/323761, MAP:0.101
testing adjectivenouns, coverage:1836/1944, spearmanr:0.420, p:0.000
testing verbobjects,    coverage:1836/1944, spearmanr:0.338, p:0.000
testing compoundnouns,  coverage:1782/1944, spearmanr:0.487, p:0.000
testing all,            coverage:5454/5832, spearmanr:0.432, p:0.000
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
For visualisation, if pick_random is passed as False, the first n items will be selected for the reduction of the space.
```python
sapce.plot_space(method='tsne', word_count=1000, pick_random=True, size=(50, 50))
```
![space](https://github.com/lorenzoscottb/txt2space/blob/master/tests/semsp_test.png)

## Requirments
 - numpy >= 1.15.4
 - nltk >= 3.3
 - sklearn >= 0.20.0
