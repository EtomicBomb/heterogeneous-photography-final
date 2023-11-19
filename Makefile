ccflags = -g -Ofast --shared -fPIC -Wall
nvccflags = -arch=sm_60 -g -O3 --shared -Xcompiler -fPIC -Xcompiler -Wall

run-cpu: target/cpu.so
	SHARED_OBJECT_PATH=$< python3 src/main.py

run-gpu: target/gpu.so
	SHARED_OBJECT_PATH=$< python3 src/main.py

target/cpu.so: src/cpu.cpp
	g++ $< ${ccflags} -o $@

target/gpu.so: src/gpu.cu
	nvcc $< ${nvccflags} -o $@

clean:
	rm -f target/*

.PHONY: run-cpu run-gpu clean

