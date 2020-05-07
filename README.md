# IR_utils



## Datasets

* [WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) - Microsoft Learning to Rank Datasets 
* [LETOR](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) - Learning to Rank for Information Retrieval 
* [Yahoo](http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf) - Yahoo! Learning to Rank Challenge Overview

## Useful Libs (Python)

* [DEAP](https://github.com/deap) - Distributed Evolutionary Algorithms in Python
* [cuML](https://github.com/rapidsai/cuml) - RAPIDS Machine Learning Library 
* [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library
* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine Learning in Python




## Installation R + rpy2
```
Para instalar uma versão do R acima da 3.6
```

sudo apt-get remove r-base-core
sudo apt-get remove r-base
sudo apt-get autoremove


sudo apt install dirmngr apt-transport-https ca-certificates software-properties-common gnupg2
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
sudo apt update
sudo apt install r-base
```
```
sudo apt install python3-pip 
```
```
pip3 install rpy2
```
