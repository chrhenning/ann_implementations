# TODOs

* let dataset classes use the logger (instead of printing to console)
* Write a dataset wrapper, that allows usage of sharded data (HDF5)
* write handlers for the following datasets
  * [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
  * [LSUN (bedroom)](http://lsun.cs.princeton.edu/2017/)
  * [SVHN](http://ufldl.stanford.edu/housenumbers/)
  * [STL-10](https://cs.stanford.edu/~acoates/stl10/)
  * [affNist](https://www.cs.toronto.edu/~tijmen/affNIST/)
  * [smallNorb](https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/)
* Allow to distinguish between multi-label classification and single-label classification.
* Build a python package, that can be used within our group (or in genereal): [see here](https://packaging.python.org/tutorials/packaging-projects/)
  * We should put some basic code into the `__init__.py` files, such that we already check the output folder, when the package is loaded and initialize the logger ([see here](http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html)).
