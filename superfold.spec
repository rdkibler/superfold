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
apt-get install -y libx11-6 libxau6 libxext6 libxrender1 libgl1-mesa-glx
#apt-get install -y git build-essential
#apt-get install -y vim
apt-get clean

# Download superfold
apt-get install -y git
git clone https://github.com/rdkibler/superfold.git /opt/superfold

# Download and link to the alphafold weights
mkdir -p /opt/weights
mkdir -p /opt/weights/params
apt-get install -y wget tar
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -P /opt/weights/
tar -xvf /opt/weights/alphafold_params_2022-12-06.tar -C /opt/weights/params/
rm /opt/weights/alphafold_params_2022-12-06.tar
echo /opt/weights/ > /opt/superfold/alphafold_weights.pth

#bash /opt/miniconda.sh -b -u -p /usr
#rm /opt/miniconda.sh
rm -rf /usr/lib/terminfo
export MAMBA_ROOT_PREFIX=/usr
export MAMBA_EXE="/opt/micromamba";
eval "$(/opt/micromamba shell hook -s posix)"
export CONDA_OVERRIDE_CUDA=12

micromamba install -p /usr \
    -c pyg \
    -c pytorch \
    -c schrodinger \
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
    conda \
    conda-package-handling \
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
    mamba \
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
    pymol \
    pymol-bundle \
    pynvim \
    pynvml \
    python=3.8 \
    python-blosc \
    python-dateutil \
    python-graphviz \
    pytorch=*=*cuda*cudnn* \
    pytorch-cluster=*=*cu* \
    pytorch-lightning \
    pytorch-mutex=*=*cu* \
    pytorch-scatter=*=*cu* \
    pytorch-sparse=*=*cu* \
    pytorch-spline-conv=*=*cu* \
    pytorch-cuda=11.7 \
    rdkit \
    regex \
    requests \
    rsa \
    ruby \
    scikit-learn \
    scipy \
    send2trash \
    setuptools \
    simpervisor \
    sympy \
    six \
    statsmodels \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorflow \
    tensorflow-estimator \
    deepmodeling::tensorflow-io-gcs-filesystem \
    termcolor \
    toolz \
    torchaudio=*=*cu* \
    torchvision=*=*cu* \
    tqdm \
    traitlets \
    traittypes \
    typed-ast \
    typing-extensions \
    wandb \
    wheel \
    widgetsnbextension \
    wrapt \
    yt \
    zict \
    omegaconf \
    hydra-core \
    ipdb \
    deepdiff \
    e3nn \
    deepspeed \
    ml-collections \
    assertpy \
    python-dateutil \
    pyrsistent \
    mysql-connector-python \
    pdbfixer \
    cuda-nvcc



pip install ml-collections
pip install dm-haiku
pip install opt_einsum 


# Clean up
micromamba clean -a -y
pip cache purge

%environment 
export PATH=$PATH:/usr/local/cuda/bin
 
%runscript
exec python /opt/superfold/run_superfold.py "$@"