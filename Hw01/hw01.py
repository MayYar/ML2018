
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


N = 3   #the number of polynomial bases
Lambda = 1.0
input_file = "data.csv"

test = [[1,12], [122,34], [-12,323], [2,3]]
test2 = [[1,12], [122,34], [-12,323]]

a = [[1, 2, 3], [4, 5, 6]]
b = [[1, 2], [3, 4], [5, 6]]
def tranpose_matrix(matrix):
    row = len(matrix)
    col = len(matrix[0])
    t = []
    #transpose row and column
    for i in range(col):
        t.append([matrix[j][i] for j in range(row)])

    return t

def mul_matrix(a,b):
    if len(a[0]) != len(b):
        raise Exception("mul matrix Error") 
    
    np_a = np.array(a)
    np_b = np.array(b)
    m = np.dot(np_a,np_b)

    return m

def add_matrix(a,b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise Exception("add matrix Error")    

    c = []
    for i in range(len(a)):
        c.append([a[i][j] + b[i][j] for j in range(len(a[0]))])

    return c

def uni_matrix(n):
    u = []
    for i in range(n):
        u.append([1 if i == j else 0 for j in range(n)])

    return u

def matrix_scale(n, a):
    t = []
    for i in a:
        t.append([j*n for j in i])

    return t

print(uni_matrix(2))
# def LU_decomposition():

# Use LU decomposition to find the inverse of (ATA + lambda*I)

def solve_X_by_LU(L, U, y):
    x = [0 for _ in range(len(y))];
    Ux = [];
    for i in range(len(y)):
        tmp = y[i]
        for j in range(i):
            tmp = tmp - Ux[j]*L[i][j]
        Ux.append(tmp)
    for i in reversed(range(len(Ux))):
        tmp = Ux[i]
        for j in range(len(Ux) - i - 1):
            j = j + i + 1
            tmp = tmp - x[j]*U[i][j] 
        x[i] = tmp / U[i][i]
        
    return x;
    
def inverse_matrix(m):
    row = len(m)
    col = len(m[0])
    if (row != col):
        return None;
    L = unit_matrix(row);
    U = [list(m[i]) for i in range(row)];
    for i in range(row):
        for j in range(col):
            if j >= i:
                continue;
            L[i][j] = U[i][j]/U[j][j];
            U[i] = list(map(lambda x,y:x - (L[i][j] * y) , U[i], U[j]))
    
    b = unit_matrix(row)
    inverse_m = [];
    for i in b:
        inverse_m.append(solve_X_by_LU(L, U, i))
    inverse_m = tranpose_matrix(inverse_m)
    return inverse_m;


# find the inverse of (ATA + lambda*I)
def linear_regression(datas, N = 2, Lambda = 1.0):
    # A,b = convert_data_Ab(datas, N)
    
    ATA = mul_matrix(tranpose_matrix(A), A)
    ATA_Lambda = add_matrix(ATA, matrix_scale(Lambda, unit_matrix(N)))
    x_vector = mul_matrix(inverse_matrix(ATA_Lambda), tranpose_matrix(A))
    x_vector = mul_matrix(x_vector, b)
    
    return x_vector;

# print(tranpose_matrix(test))
#load data and show data points
input_data = pd.read_csv(input_file, header = None, names = ['x','y'])
# print(input_data)
# plt.plot(input_data['x'], input_data['y'], 'r.')
# plt.show()