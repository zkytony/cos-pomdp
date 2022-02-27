#!/bin/bash
# Run this script from repository root
if [[ ! $PWD = *cos-pomdp ]]; then
    echo "You must be in the root directory of the cos-pomdp repository."
    return 1
fi
repo_root=$PWD

if [ ! -d "venv/cosp" ]; then
    virtualenv -p python3 venv/cosp
    source venv/cosp/bin/activate
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

# Submodules
cd $repo_root
if [ $update_submodules = true ]; then
    git submodule update --force --recursive --init --remote
fi

# create symbolic link to mjolnir directory
if [ ! -e "cospomdp_apps/thor/mjolnir" ]; then
    ln -sf $(readlink -f external/mjolnir) cospomdp_apps/thor/mjolnir
fi

# clone and install necessary repositories, if not yet already
if [ $setup_repos = true ]; then
    pip install numpy
    pip install matplotlib
    pip install torchvision
    pip install networkx
    pip install pytest
    pip install sciex

    cd external
    # clone thortils
    if [ ! -d "thortils" ]; then
        git clone git@github.com:zkytony/thortils.git
        cd thortils
        git checkout v3.3.4-stable
        git pull
        pip install -e .
        cd ..
    fi
fi
cd $repo_root

# ask if want to create alias command
if [[ $source_venv = false ]]; then
    read -p "Create alias 'cosp' for starting cos-pomdp venv? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]];
    then
        echo -e "alias cosp='cd $repo_root; source setup.bash'" >> ~/.bashrc
    fi
fi
