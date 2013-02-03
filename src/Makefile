all: main.cu
	nvcc -O2 -I/usr/share/R/include -Xcompiler "-fpic" -c main.cu
	nvcc -shared -lcublas main.o -o GPULassoPath.so

clean:
	rm GPULassoPath.so
	rm main.o
