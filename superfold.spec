Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup

%files
/etc/localtime
/etc/apt/sources.list
/etc/ssl/certs/ca-certificates.crt
/home/cdemakis/apptainer/files/bin/micromamba /opt/micromamba
#/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh 

%post
rm /bin/sh; ln -s /bin/bash /bin/sh

ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
apt-get install -y libx11-6 libxau6 libxext6 g++ wget tar
#apt-get install -y git build-essential
#apt-get install -y vim

bash /opt/miniconda.sh -b -u -p /usr
rm /opt/miniconda.sh
rm -rf /usr/lib/terminfo


conda install -p /usr \
    -c pyg \
    -c pytorch \
    -c dglteam/label/cu117 \
    -c conda-forge \
    -c bioconda \
    -c nvidia \
    abseil-cpp \
    absl-py \
    atk-1.0 \
    attrs \
    billiard \
    binutils_impl_linux-64 \
    biopython \
    blas \
    blosc \
    bokeh \
    chex \
    clang-11 \
    contextlib2 \
    cryptography \
    cudatoolkit \
    cudnn \
    dataclasses \
    decorator \
    dgl \
    dglteam::dgl-cuda11.7 \
    distributed \
    dm-tree \
    flametree \
    flask \
    flatbuffers \
    git \
    gitpython \
    graphviz \
    h5py \
    holoviews \
    icecream \
    idna \
    immutabledict \
    ipykernel \
    ipympl \
    ipython \
    ipython_genutils \
    isort \
    jax \
    jaxlib=*=*cuda*py38* \
    keras \
    markdown \
    matplotlib \
    mock \
    more-itertools \
    nb_black \
    nbclassic \
    nbclient \
    nbconvert \
    nbdime \
    nbformat \
    numba \
    numdifftools \
    numpy=1.23.5 \
    nvidia-apex \
    openbabel \
    openmm=7.5.1 \
    openpyxl \
    pandas \
    pandoc \
    parallel \
    perl \
    pip \
    proglog \
    psutil \
    pybind11 \
    pyg=*=*cu* \
    pymatgen \
    python=3.8 \
    rsa \
    setuptools \
    tensorflow \
    tensorflow-estimator \
    tqdm \
    typing-extensions \
    wheel \
    wrapt \
    yt \
    zict \
    omegaconf \
    ipdb \
    deepdiff \
    e3nn \
    deepspeed \
    ml-collections \
    assertpy \
    python-dateutil \
    pyrsistent \
    cuda-nvcc

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
