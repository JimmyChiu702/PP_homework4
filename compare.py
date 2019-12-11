with open('cuda_result.txt', 'r') as f:
    cuda_result = f.read().splitlines()
    cuda_result = list(map(float, cuda_result))

with open('serial_result.txt', 'r') as f:
    serial_result = f.read().splitlines()
    serial_result = list(map(float, serial_result))

err = ['{} {} {}'.format(serial_result[i], cuda_result[i], abs(serial_result[i]-cuda_result[i])) for i in range(len(cuda_result)) if abs(serial_result[i]-cuda_result[i]) > 0.001]
# for e in err:
#     print(e)
print('Total error: {}'.format(len(err)))
print('Error rate: {}'.format(len(err)/len(cuda_result)))