Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup


%files
/etc/localtime
/etc/apt/sources.list
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
apt-get install -y python3.9 python3-pip 
apt-get install -y pymol
pip install absl-py dm-tree tensorflow ml-collections tqdm
pip install "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dm-haiku
pip install psutil



# Download and link to the alphafold weights
mkdir -p /opt/weights
apt-get install -y wget tar
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -P /opt/weights/
tar -xvf /opt/weights/alphafold_params_2022-12-06.tar -C /opt/weights/
rm /opt/weights/alphafold_params_2022-12-06.tar
echo /opt/weights/ > /opt/superfold/alphafold_weights.pth

# Clean up
apt-get -y purge git
apt-get -y autoremove
apt-get clean

%environment
export PATH=$PATH:/usr/local/cuda/bin
export PYTHONNOUSERSITE=1

%runscript
exec python3 "$@"
#python /opt/superfold/run_superfold.py "$@"


%help
https://github.com/rdkibler/superfold
