######################
# Tried and didn't work; Got
#  "nvcc fatal   : Unsupported gpu architecture 'compute_30'"
# When compiling darknet. Also numerous errors when installing other stuff
#
# Also, tensorflow 1.5.0 has been removed from pypi:
# "Could not find a version that satisfies the requirement tensorflow-gpu==1.5.0"
first_time=false
cd methods
if [ ! -d ./iqa ];
then
    mkdir iqa
    cd iqa
    git clone git@github.com:danielgordon10/thor-iqa-cvpr-2018.git
    first_time=true
    cd ..
fi

cd iqa/thor-iqa-cvpr-2018
virtualenv -p $(which python3) thor_iqa_env
source thor_iqa/bin/activate

if $first_time;
then
    pip install --upgrade pip
    pip install -r requirements.txt
    sh download_weights.sh

    # Download darknet yolov3; https://pjreddie.com/darknet/install/
    git clone https://github.com/pjreddie/darknet.git
    cd darknet
    echo "You have CUDA. Change GPU=1 in Makefile"
    emacs Makefile
    make
    ./darknet

    # Add darknet python module to pythonpath; get path to site-packages
    # see this: https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv/47184788#47184788
    cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
    echo $(pwd)/darknet > "darknet.pth"

fi

# Run some tests
if $first_time;
then
    python run_thor_tests.py
fi
