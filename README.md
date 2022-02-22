# Grammar2PDDL

This package takes a data science grammar and provides code to explore the set of possible pipelines from the grammar. Features:

* Produces multiples executable [LALE](https://github.com/IBM/lale) pipelines from the grammar, with optional user constraints. It does so using AI planning.
* Trains hyperparameters and evaluates generated pipelines.
* Can use measured pipeline accuracy to produce better pipelines in subsequent iterations.

The full details are in Katz, M., Ram, P., Sohrabi, S., & Udrea, O. (2020). *Exploring Context-Free Languages via Planning: The Case for Automating Machine Learning*. Proceedings of the International Conference on Automated Planning and Scheduling, 30(1), 403-411. [PDF](https://ojs.aaai.org//index.php/ICAPS/article/view/6686)

## Installation


### 1. Install [Singularity](https://sylabs.io/singularity/) (needed for planutils)
  * Currently, singularity doesn't support MAC OS.
  * Checkout [Admin guide](https://sylabs.io/guides/3.7/admin-guide/index.html) for the details
```
## Install system dependencies in Debian/Ubuntu
$ sudo apt-get update && sudo apt-get install -y \
    build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup-bin

## Install system dependencies in Centos/Redhat    
$ sudo yum update -y && \
     sudo yum groupinstall -y 'Development Tools' && \
     sudo yum install -y \
     openssl-devel \
     libuuid-devel \
     libseccomp-devel \
     wget \
     squashfs-tools \
     cryptsetup

## Install Go
$ export VERSION=1.14.12 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz
$ echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
    source ~/.bashrc   
    
## Download and build singularity
$ wget https://github.com/hpcng/singularity/releases/download/v3.7.2/singularity-3.7.2.tar.gz
$ tar xvf singularity-3.7.2.tar.gz
$ cd singularity &&   ./mconfig &&   cd ./builddir &&   make &&   sudo make install && cd ../..   
```

### 2. Install Python dependencies in conda environment
  * Install packages through pip
  * Setup [planutils](https://github.com/AI-Planning/planutils) and install [K* planner](https://github.com/ctpelok77/kstar) and [HTN to PDDL](https://github.com/ronwalf/HTN-Translation) translator.
```
## Create a conda environment
$ conda create -n grammar2plans python=3.7
$ conda activate grammar2plans

## Install python packages
$ pip install -r requirements.txt

## Setup planutils
$ planutils setup
$ export PATH=$PATH:~/.planutils/bin
$ planutils install kstar
$ planutils install hpddl2pddl
```    


## Getting started/samples

1. Start jupyter notebooks: `$ jupyter notebook`.
2. Navigate to `notebooks/DataSciencePipelinePlanningTutorial`
3. Executing the cells will create intermediate planning and result files in `output`. 
  * You can run an instance of [VS Code](https://code.visualstudio.com/) with the [PDDL language support plugin](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl) to see intermediate planning task files: `code output/`


## Using alternate planners

By default, the code using the `kstar` planner that is part of the `planutils` package. You can however use a different planner by setting the `PLANNER_URL` environment variable to a service with a matching REST API.

1. Download the [IBM AI Planner Service](https://github.com/IBM/AIPlanningService) or any other service with the same REST API.
2. Run the service - as a local docker container or as part of a cloud service.
3. Set `PLANNER_URL` to the planner you want to use. For instance, if you run the service in a local docker container as per the service README and you would like to use `kstar`, you would set `PLANNER_URL=http://localhost:4501/planners/topk/kstar-topk`.
4. To return to using the `planutils` version of `kstar`, `unset PLANNER_URL`.
