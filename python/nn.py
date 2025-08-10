import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forword(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forword(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis = 0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x. W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)

        self.grads[0][...] = dW
        return dx
    
class LayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # initialize weights
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)

        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forword(x)
        return x



### Run Example    
x = np.random.randn(10, 2)
print(x.shape)
print(x.ndim)

model = LayerNet(2, 4, 3)
s = model.predict(x)
# print(s)
print(s.shape)
print(s.ndim)
