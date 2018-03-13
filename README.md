# node2vec-c

[node2vec](http://snap.stanford.edu/node2vec/) implementation in dependency-less C++. node2vec uses short biased random walks to learn representations for vertices in unweighted graphs.
Other implementations are available in [C++ in SNAP project](https://github.com/snap-stanford/snap/tree/master/examples/node2vec) and a reference one in [Python + Gensim](https://github.com/aditya-grover/node2vec).

This code was developed to be used in out paper, [VERSE](https://github.com/xgfs/verse) for a fair performance comparison. Moreover, the high-performance implementation in SNAP did not match the performance of the original implementation (as for 2017).

## Installation and usage

For the executable:

    make

should be enough on most platforms. If you need to change the default compiler (i.e. to Intel), use:

    make CXX=icpc

Intel® FMA availability is crucial for performance of the implementation, meaning the processor  Haswell (2013). You will get a warning on runtime if your processor does not support it.

### Usage

```
Usage: node2vec [OPTIONS]

Options:
  -input PATH                    Input file in binary CSR format
  -output PATH                   Output file, written in binary
  -threads INT                   Number of threads to use (default 1)
                                   Note: hyperthreading helps as well
  -dim INT                       node2vec parameter d: dimensionality of
                                   embeddings (default 128)
  -nwalks INT                    node2vec parameter gamma: number of walks per
                                   node (default 80)
  -walklen INT                   node2vec parameter t: length of random walk
                                   from each node(default 80)
  -window INT                    node2vec parameter w: window size (default 10)
  -nsamples INT                  node2vec parameter k: number of negative samples (default 5)
  -p FLOAT                       node2vec parameter p: random walk bias (default 1)
  -q FLOAT                       node2vec parameter q: random walk bias (default 1)
  -lr FLOAT                      Initial learning rate
  -seed INT                      Sets the random number generator seed to INT
  -verbose INT                   Controls verbosity level in [0,1,2], 0 meaning
                                   nothing will be displayed, and 2 mening
                                   training progress will be displayed.
```

### Graph format

This implementation uses a custom graph format, namely binary [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) (BCSR) format for efficiency and reduced memory usage. Converter for three common graph formats (MATLAB sparse matrix, adjacency list, edge list) can be found in the root directory of [our main code repository](https://github.com/xgfs/verse).

## Citing

If you find node2vec useful in your research, we ask that you cite the original paper:

    @inproceedings{Grover:2016:NSF:2939672.2939754,
        author = {Grover, Aditya and Leskovec, Jure},
        title = {Node2Vec: Scalable Feature Learning for Networks},
        booktitle = {Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
        series = {KDD '16},
        year = {2016},
        isbn = {978-1-4503-4232-2},
        location = {San Francisco, California, USA},
        pages = {855--864},
        numpages = {10},
        url = {http://doi.acm.org/10.1145/2939672.2939754},
        doi = {10.1145/2939672.2939754},
        acmid = {2939754},
        publisher = {ACM},
        address = {New York, NY, USA},
        keywords = {feature learning, graph representations, information networks, node embeddings},
    } 

## Contact

`echo "%7=87.=<2=<>527@192.()" | tr '#-)/->' '_-|'`