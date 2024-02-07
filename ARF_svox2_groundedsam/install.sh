# Remove pytorch, torchvision and cudatoolkit from environment.yml
conda env create -f environment.yml
conda activate arf-svox2
# FOR CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# FOR CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
pip install ninja
pip install -e . --verbose
pip install icecream
pip install huggingface-hub
pip install diffusers
pip install segment-anything
pip install lama-cleaner
# DOWNLOAD SAM MODEL & PLACE IN /opt/ https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth
# COCO LABELS: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/