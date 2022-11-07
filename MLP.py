import numpy as np

class MLP:

    def __init__(self, number_of_input_units, number_of_hidden_units, number_of_output_units, eta,w_init_min=0, w_init_max=100):
        self.ius=number_of_input_units
        self.hus=number_of_hidden_units
        self.ous=number_of_output_units
        self.eta=eta
        self.W_hi= np.random.randint(low=w_init_min,high=w_init_max,size=(self.hus,self.ius))/100
        self.w_h0= np.random.randint(low=w_init_min,high=w_init_max,size=(self.hus))/100
        self.W_oh= np.random.randint(low=w_init_min,high=w_init_max,size=(self.ous,self.hus))/100
        self.w_o0= np.random.randint(low=w_init_min,high=w_init_max,size=(self.ous))/100
        print("W_hi:",self.W_hi, "W_oh",self.W_oh)
        print("w_h0",self.w_h0, "w_o0", self.w_o0)

    def h_x(self, x):
        o_h= self.W_hi.dot(x) + self.w_h0
        o_k=self.W_oh.dot(o_h) + self.w_o0
        return o_k


NN=MLP(3,2,1,0.5)
x_1=np.array([1,3,5])
print(NN.h_x(x_1))