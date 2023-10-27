Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup


%files
/etc/localtime
/etc/apt/sources.list
/home/cdemakis/apptainer/files/bin/micromamba /opt/micromamba
#/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
apt-get clean

# Download superfold
apt-get install -y git
git clone https://github.com/rdkibler/superfold.git /opt/superfold

# Install python packages
rm -rf /usr/lib/terminfo
export MAMBA_ROOT_PREFIX=/usr
export MAMBA_EXE="/opt/micromamba";
eval "$(/opt/micromamba shell hook -s posix)"
export CONDA_OVERRIDE_CUDA=12
#bash /opt/miniconda.sh -b -u -p /usr
#conda install \
micromamba install -p /usr \
    -c pyg \
    -c pytorch \
    -c schrodinger \
    -c dglteam/label/cu117 \
    -c conda-forge \
    -c bioconda \
    -c nvidia \
    python=3.9 \
    cudatoolkit=11.4 \
    tensorflow \
    cudnn=8.2 \
    pip \
    numpy=1.23.5 \
    scipy \
    pandas \
    biopython=1.78 \
    psutil \
    tqdm \
    absl-py \
    dm-tree \
    immutabledict \
    chex \
    pymol \
    jax \
    jaxlib=*=*cuda*py39* \

pip install ml-collections dm-haiku


# Download and link to the alphafold weights
mkdir -p /opt/weights
apt-get install -y wget tar
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -P /opt/weights/
tar -xvf /opt/weights/alphafold_params_2022-12-06.tar -C /opt/weights/
rm /opt/weights/alphafold_params_2022-12-06.tar
echo /opt/weights/ > /opt/superfold/alphafold_weights.pth

# Clean up
#conda clean -a -y
micromamba clean -a -y
apt-get -y purge git
apt-get -y autoremove
apt-get clean
#rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin
export PYTHONNOUSERSITE=1

%runscript
exec python3 "$@"
#python /opt/superfold/run_superfold.py "$@"


%help
https://github.com/rdkibler/superfold
