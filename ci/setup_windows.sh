CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe
./cuda.exe -s nvcc_12.2 cudart_12.2
rm cuda.exe