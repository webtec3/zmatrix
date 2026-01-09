GPU_OBJ = gpu_kernels.o

# Compila gpu_kernels.cu com nvcc
$(GPU_OBJ): src/gpu_kernels.cu
	$(NVCC) $(ZMATRIX_NVCCFLAGS) --compile -o $@ $< $(ZMATRIX_CPPFLAGS)

# Garante que será linkado com a extensão
shared_objects += $(GPU_OBJ)