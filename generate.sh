mkdir -p generated
nvcc -O3 -ptx -arch=sm_50 csrc/optixdev.cu -Iinclude -o generated/torchoptixdev.ptx
xxd -i generated/torchoptixdev.ptx >csrc/optixdevptx.h
