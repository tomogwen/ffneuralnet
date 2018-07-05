
import numpy as np
import matplotlib.pyplot as plt

x = np.array([ [1, 1, 1, 1, 1, 1, 0],  # 0
               [0, 1, 1, 0, 0, 0, 0],  # 1
               [1, 1, 0, 1, 1, 0, 1],  # 2
               [1, 1, 1, 1, 0, 0, 1],  # 3
               [0, 1, 1, 0, 0, 1, 1],  # 4
               [1, 0, 1, 1, 0, 1, 1],  # 5
               [0, 0, 1, 1, 1, 1, 1],  # 6
               [1, 1, 1, 0, 0, 0, 0],  # 7
               [1, 1, 1, 1, 1, 1, 1],  # 8
               [1, 1, 1, 0, 0, 1, 1]]) # 9


y = np.array( [[ 0, 0, 0, 0 ],
               [ 0, 0, 0, 1 ],
               [ 0, 0, 1, 0 ],
               [ 0, 0, 1, 1 ],
               [ 0, 1, 0, 0 ],
               [ 0, 1, 0, 1 ],
               [ 0, 1, 1, 0 ],
               [ 0, 1, 1, 1 ],
               [ 1, 0, 0, 0 ],
               [ 1, 0, 0, 1 ]])


numInput       = 7
numHiddenNodes = 80
numOutput      = 4
sampleSize     = 10
eta            = 0.01  # learning rate
lam            = 0.01  # regularisation rate

W1 = np.random.rand(numInput, numHiddenNodes)
W2 = np.random.rand(numHiddenNodes, numOutput)
b1 = np.zeros((1, numHiddenNodes))
b2 = np.zeros((1, numOutput))


def softmax(z2):
    ex = np.exp(z2)
    return ex/np.sum(ex)


def predict(xi):
    a1 = np.tanh( xi.dot(W1) + b1)
    #ybar = softmax(a1.dot(W2) + b2)
    ybar = np.tanh(a1.dot(W2) + b2)
    return a1, ybar


def loss(y, ybar, sampleSize):
    # return -1/sampleSize * np.sum(np.multiply(y, np.log(ybar)))
    return np.sqrt(np.sum((y-ybar))**2)


def backprop(xj, ybar, yj, a1):
    del3 = ybar - np.transpose(yj)
    del2 = (1 - a1 ** 2) * (del3.dot(np.transpose(W2)))
    dW2  = np.matmul(np.transpose(a1), del3)
    dW1  = np.reshape(np.transpose(xj), (7,1)).dot(del2)
    return dW2, del3, dW1, del2


if __name__ == "__main__":
    results = []
    for i in range(10000):
        for j in range(0,10):
            a1, ybar = predict(np.transpose(x[j]))
            dW2, db2, dW1, db1 = backprop(np.transpose(x[j]), ybar, np.transpose(y[j]), a1)

            dW2 += lam * W2
            dW1 += lam * W1

            W1 += -eta * dW1
            b1 += -eta * db1
            W2 += -eta * dW2
            b2 += -eta * db2

        if i % 100 == 0:
            t, ybar = predict(x[j])
            results.append(loss(y[j], ybar[0], sampleSize))
            print i

    plt.plot(results)
    plt.show()
