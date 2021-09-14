all:
	nvcc -std=c++11 -c -I./ layer.cu main.cu nn.cu  
	nvcc -lcublas main.o nn.o layer.o