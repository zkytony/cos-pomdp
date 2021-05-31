cd methods
mkdir MJOLNIR
cd MJOLNIR

virtualenv -p $(which python3) mjolnir_env
source mjolnir_env/bin/activate
git clone https://github.com/zkytony/MJOLNIR.git

cd MJOLNIR
pip install -r requirements.txt
pip install networkx
pip install ai2thor==1.0.1
pip install h5py
pip install scipy
pip install torchvision==0.2.2.post3
pip install Pillow
pip install tensorboardX
pip install torch==1.8.1

# data.zip
gdown https://drive.google.com/uc?id=1yWWzgcsT1PtFJdXLPaozvE1av3yaZoX0
