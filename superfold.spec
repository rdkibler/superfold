Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup

%files
/etc/localtime
/etc/hosts
/etc/apt/sources.list
/etc/ssl/certs/ca-certificates.crt
# /home/cdemakis/apptainer/files/bin/micromamba /opt/micromamba
/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh 

%post
rm /bin/sh; ln -s /bin/bash /bin/sh

ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
apt-get install -y libx11-6 libxau6 libxext6 g++ wget tar
apt-get install -y git build-essential
#apt-get install -y vim

bash /opt/miniconda.sh -b -u -p /usr
conda update conda
rm /opt/miniconda.sh
rm -rf /usr/lib/terminfo

conda install -p /usr \
    -c conda-forge \
    python=3.8 \
    pip=20.2.4

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


conda install -p /usr \
    -c pyg \
    -c pytorch \
    -c conda-forge \
    -c bioconda \
    -c nvidia \
    cudatoolkit \
    cudnn \
    cuda-nvcc \
    tensorflow \


conda install -p /usr \
    -c pyg \
    -c pytorch \
    -c conda-forge \
    -c bioconda \
    -c nvidia \
    attrs \
    biopython \
    blas \
    blosc \
    bokeh \
    cryptography \
    dataclasses \
    decorator \
    distributed \
    dm-tree \
    h5py \
    idna \
    ipykernel \
    ipython \
    ipython_genutils \
    matplotlib \
    mock \
    more-itertools \
    numpy=1.23.5 \
    pandas \
    pandoc \
    pip \
    psutil \
    setuptools \
    tqdm \
    typing-extensions \
    wheel \
    zict \
    ml-collections \
    python-dateutil \
    pyrsistent 

pip install dm-haiku
pip install opt_einsum 

# Download superfold
git clone https://github.com/rdkibler/superfold.git /opt/superfold

#compile mmalign
here=$(pwd)
cd /opt/superfold/mmalign
g++ -static -O3 -ffast-math -lm -o MMalign MMalign.cpp
cd $here

# Download and link to the alphafold weights
mkdir -p /opt/weights
mkdir -p /opt/weights/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -P /opt/weights/
tar -xvf /opt/weights/alphafold_params_2022-12-06.tar -C /opt/weights/params/
rm /opt/weights/alphafold_params_2022-12-06.tar
echo /opt/weights/ > /opt/superfold/alphafold_weights.pth

# Clean up
apt-get clean
#micromamba clean -a -y
conda clean -a -y
pip cache purge

%environment 
export PATH=$PATH:/usr/local/cuda/bin
 
%runscript
exec python /opt/superfold/run_superfold.py "$@"
