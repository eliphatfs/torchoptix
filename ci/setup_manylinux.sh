set -e
set -x

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
# error mirrorlist.centos.org doesn't exists anymore.
sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
yum install --setopt=obsoletes=0 -y \
    cuda-driver-devel-11-8-11.8.89-1 \
    cuda-cudart-devel-11-8-11.8.89-1 \
    cuda-nvcc-11-8-11.8.89-1
touch /usr/local/cuda/lib64/libcuda.so.1
chmod +x /usr/local/cuda/lib64/libcuda.so.1
