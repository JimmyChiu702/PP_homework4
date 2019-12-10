all:
	nvcc cuda_wave.cu -o cuda_wave
clean:
	rm -f serial_wave
