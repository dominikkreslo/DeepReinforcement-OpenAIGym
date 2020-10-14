import numpy as np

a = []
x = np.array([.63, .37])
num_zero = 0
num_one = 0

for i in range(100):
    rndNum = np.random.choice(2,p=x)
    if rndNum == 0:
        num_zero += 1
    elif rndNum == 1:
        num_one += 1
    a.append(rndNum)

print(a)
print("Number of Zeros: " + str(num_zero))
print("Number of Ones: " + str(num_one))