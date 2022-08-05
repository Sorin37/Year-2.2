import matplotlib.pyplot as plt
import numpy as np

iterations = 100
learning_rate=0.1


def h(x, theta):
    return np.matmul(x, theta.transpose())

def scaleMatrix(mat):
    max= np.max(mat, axis=0)
    min= np.min(mat, axis=0)
    dif=max-min
    if 0 in dif:
        dif[np.where(dif==0)]=1
    return (mat-min)/dif, min, max

def appendColumn(mat):
    return np.insert(mat,0,1,axis=1)

def readFile(filename):
    data = np.genfromtxt(filename, delimiter=";", skip_header=1)
    return data

def cost(x,y,theta):
    predict=h(x,theta)
    newCost= np.sum((predict- y)**2)/(2*len(y))
    return newCost


def gradient_descent(x, y):
    x,min,max= scaleMatrix(x)
    x= appendColumn(x)
    theta= np.zeros(x.shape[1])
    newcost= np.zeros(iterations)
    for i in range(iterations):
        predict=h(x, theta)
        theta=theta - learning_rate*sum(np.matmul((predict-y),x))/len(y)
        newcost[i] = cost(x,y,theta)
    return newcost,theta




def algMethod(x,y):
    x=appendColumn(x)
    theta= np.matmul(np.linalg.pinv(x),y)
    h= np.matmul(x,theta.transpose())
    myCost= cost(x,y,theta)
    for i in range(len(x)):
        print( "Valoare prezisa:", h[i], "Valoare efectiva", y[i])
    print("Cost:", myCost)
    return theta


data = readFile('winequality-white.csv')

shuffle_data = data.copy()
np.random.shuffle(shuffle_data)

# x= data[:,:-1].copy()
# y= data[:,-1].copy()

data_train=shuffle_data[:int(len(shuffle_data)*0.7)]
data_test=shuffle_data[int(len(shuffle_data)*0.7):]

x_train= data_train[:,:-1].copy()
y_train= data_train[:,-1].copy()



x_test= data_test[:,:-1].copy()
y_test= data_test[:,-1].copy()


gd_costs, train_theta=(gradient_descent(x_train,y_train))
plt.plot(np.arange(iterations),gd_costs)
plt.show()


x_test_gd,min,max=scaleMatrix(x_test)
x_test_gd=appendColumn(x_test_gd)
cost_for_GD= cost(x_test_gd,y_test,train_theta)
print("Test cost for gradient descent", cost_for_GD)


train_theta2= algMethod(x_train,y_train)
x_test_alg=appendColumn(x_test)
cost_for_algMeth= cost(x_test_alg,y_test,train_theta2)
print("Cost for algebric method", cost_for_algMeth)
# algMethod(x_train,y_train)



# theta= np.matmul(np.linalg.pinv(x_train),y_train)
# myCost= cost(x_test,y_test,theta)
# print("Cost test",myCost)

