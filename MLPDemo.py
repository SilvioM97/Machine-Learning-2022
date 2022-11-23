from MLP import *

mlp=MLP([2,3,1])

TR=[([2,1],1),([1,2],1), ([1,1],0), ([0,0],0)]
mlp.training(TR, epochs=1000)
mlp.learning_curve()
