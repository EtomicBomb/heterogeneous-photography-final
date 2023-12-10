ccflags = -g -Ofast -fopenmp --shared -fPIC -Wall
nvccflags = -g -O3 -arch=sm_60 --shared -Xcompiler -fPIC -Xcompiler -Wall

run: target/cpu.so
	python3 src/main.py

run-gpu: target/cpu.so target/gpu.so
	python3 src/main.py

test: target/cpu.so target/gpu.so
	python3 -m pytest src/test_scanline_stereo.py

testing: target/cpu.so target/gpu.so
	python3 src/testing.py

target/cpu.so: src/cpu.cpp
	g++ $< ${ccflags} -o $@

target/gpu.so: src/gpu.cu
	nvcc $< ${nvccflags} -o $@

clean:
	rm -f target/*

.PHONY: run run-gpu test clean

