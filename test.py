from scipy.io import loadmat


mat = loadmat('/media/lex/ee2d4e17-32c9-4a5a-98b9-655706039d25/lexibender/data/sun397/Partitions/split10.mat')

print(mat['split'][0][0])
