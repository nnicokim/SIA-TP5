NVCC=nvcc
CFLAGS=-Xcompiler -fPIC -shared -O3

all: libbackend.so

libbackend.so: backend.cu
	$(NVCC) $(CFLAGS) -o libbackend.so backend.cu -lcurand

clean:
	rm -f libbackend.so
