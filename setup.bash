#!/bin/bash

cd $HOME/repo/cos-pomdp/

if [ ! -d "venv/cosp" ]; then
    virtualenv -p $(which python3) venv/cosp
    pip install -e .  # install the cos-pomdp package
fi

source_venv=true
if [[ "$VIRTUAL_ENV" == *"cosp"* ]]; then
    source_venv=false
fi

if [ $source_venv = true ]; then
    source venv/cosp/bin/activate
fi

# Parse arguments
setup_repos=false
update_submodules=false
for arg in "$@"
do
    if [[ "$arg" == -* ]]; then
        if [ "$arg" == "-I" ]; then
            setup_repos=true
        elif [ "$arg" == "-s" ]; then
            update_submodules=true
        fi
    fi
done

# clone and install necessary repositories, if not yet already
if [ $setup_repos = true ]; then
    pip install numpy
    pip install matplotlib

    # clone and install pomdp-py
    cd $HOME/repo/
    if [ ! -d "pomdp-py" ]; then
        git clone git@github.com:h2r/pomdp-py.git
        cd pomdp-py
        sudo apt install graphviz
        pip install Cython
        pip install -e .
        make build
        cd ..
    fi

    # clone thortils
    if [ ! -d "thortils" ]; then
        git clone git@github.com:zkytony/thortils.git
        cd thortils
        git checkout v3.3.4
        git pull
        pip install -e .
        cd ..
    fi

    # clone mos3d
    if [ ! -d "mos3d" ]; then
        git clone git@github.com:zkytony/mos3d-prep.git
        mv mos3d-prep mos3d
        cd mos3d
        pip install -e .
        cd ..
    fi
fi
cd $HOME/repo/cos-pomdp/

# Submodules
if [ $update_submodules = true ]; then
    git submodule update --force --recursive --init --remote
fi

# create symbolic link to mjolnir directory
if [ ! -e "cospomdp_apps/thor/mjolnir" ]; then
    ln -sf $(readlink -f external/mjolnir) cospomdp_apps/thor/mjolnir
fi
