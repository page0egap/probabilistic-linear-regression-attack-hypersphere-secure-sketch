dimension = 10
k_each_matrix = 2
num = 3

positions_shift = [[i * dimension] * k_each_matrix for i in range(num)]
print(positions_shift)
positions_shift = [item for sublist in positions_shift for item in sublist]

print(positions_shift)