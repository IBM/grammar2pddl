# Grammar2PDDL

This package takes a data science grammar and provides code to explore the set of possible pipelines from the grammar. Features:

* Produces multiples executable [LALE](https://github.com/IBM/lale) pipelines from the grammar, with optional user constraints. It does so using AI planning.
* Trains hyperparameters and evaluates generated pipelines.
* Can use measured pipeline accuracy to produce better pipelines in subsequent iterations.

The full details are in Katz, M., Ram, P., Sohrabi, S., & Udrea, O. (2020). *Exploring Context-Free Languages via Planning: The Case for Automating Machine Learning*. Proceedings of the International Conference on Automated Planning and Scheduling, 30(1), 403-411. [PDF](https://ojs.aaai.org//index.php/ICAPS/article/view/6686)

## Installation

Install or verify that you have the following pre-requisites:

1. [Docker](https://docs.docker.com/get-docker/).
   
2. [Planutils](https://github.com/AI-Planning/planutils). 
Planutils uses [Singularity](https://sylabs.io/singularity/) for individual tools. To install singularity on Debian/Ubuntu:
```
## Install system dependencies
$ sudo apt-get update &&   sudo apt-get install -y build-essential   libseccomp-dev pkg-config squashfs-tools cryptsetup

## Install Golang and set up your environment for Go
$ export VERSION=1.15.8 OS=linux ARCH=amd64  # change this as you need
$ wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz &&   sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz
 
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc &&   echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc &&   source ~/.bashrc
curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh |   sh -s -- -b $(go env GOPATH)/bin v1.21.0

## Download and build singularity

$ wget https://github.com/hpcng/singularity/releases/download/v3.7.2/singularity-3.7.2.tar.gz
$ tar xvf singularity-3.7.2.tar.gz
$ cd singularity &&   ./mconfig &&   cd ./builddir &&   make &&   sudo make install
```

Now you can install and setup planutils, and install the required tools ([K* planner](https://github.com/ctpelok77/kstar), etc. )
```
$ pip install planutils
$ planutils setup

$ planutils install kstar
```

3. [Jupyter notebooks](https://jupyter.org/install) are needed to run the sample notebook.
4. Install python requirements. We strongly recommend you do so from a `conda` environment. We have included a conda environment spec in `grammar2plans.yml`. Simply run `conda env create -f grammar2plans.yml` and then `conda activate grammar2plans`. Alternatively, you
can try installing requirments directly via `pip install -r requirements.txt`
5. Execute `./build_translator.sh` to build a docker image for the [HTN to PDDL](https://github.com/ronwalf/HTN-Translation) translator.
6. Add the code root directory to path, so that `run_translator.sh` is runnable from anywhere: `export PATH=$PATH:$(pwd)`

## Getting started/samples

1. Add the installation directory to `PYTHONPATH`: `export PYTHONPATH=$PYTHONPATH:.`
2. Start jupyter notebooks: `jupyter notebook`.
3. Navigate to `notebooks/DataSciencePipelinePlanningTutorial`
4. Executing the cells will create intermediate planning and result files in `output`. You can run an instance of [VS Code](https://code.visualstudio.com/) with the [PDDL language support plugin](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl) to see intermediate planning task files: `code output/`

## Using alternate planners

By default, the code using the `kstar` planner that is part of the `planutils` package. You can however use a different planner by setting the `PLANNER_URL` environment variable to a service with a matching REST API.

1. Download the [IBM AI Planner Service](https://github.com/IBM/AIPlanningService) or any other service with the same REST API.
2. Run the service - as a local docker container or as part of a cloud service.
3. Set `PLANNER_URL` to the planner you want to use. For instance, if you run the service in a local docker container as per the service README and you would like to use `kstar`, you would set `PLANNER_URL=http://localhost:4501/planners/topk/kstar-topk`.
4. To return to using the `planutils` version of `kstar`, `unset PLANNER_URL`.
