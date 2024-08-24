import numpy as np
import timeit 
from tinygrad import Tensor, nn
from tinygrad import Device
from tinygrad.nn.datasets import mnist
from tinygrad import TinyJit

print(Device.DEFAULT)

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor):
        x = self.l1(x).relu().max_pool2d((2,2))
        x = self.l2(x).relu().max_pool2d((2,2))
        return self.l3(x.flatten(1).dropout(0.5))        

X_train, y_train, X_test, y_test = mnist()
model = Model()
accuracy = (model(X_test).argmax(axis=1) == y_test).mean()
print(accuracy.item())
optimiser = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128

def step():
    Tensor.training = True # Allows for dropout to happen!
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], y_train[samples]
    optimiser.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optimiser.step()
    return loss

jit_step = TinyJit(step)

for step in range(7000):
    loss = jit_step()
    if step%100 == 0:
        Tensor.training = False
        accuracy = (model(X_test).argmax(axis=1) == y_test).mean().item()
        print(f"Step {step:4d}, loss {loss.item():.2f}, accuracy {accuracy*100.:.2f}%")  

# print(timeit.repeat(step, repeat=5, number=1))
 
