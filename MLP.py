import numpy as np


def sigmoid(x, a=1):
    return 1/(1 + np.exp(-(a*x)))

def E_p(o_k, d):
    return (o_k-d)**2

def E(example_list):
    #example_list in shape [(x_p,d_p),...]
    E_sum=0
    for example in example_list:
        E_sum+=E_p(example[0],example[1])
    return E_sum

def delta_k(d_k,o_k,net_k):



class MLP:

    def __init__(self, number_of_input_units, number_of_hidden_units, number_of_output_units, eta, number_of_hidden_layers=1, w_init_min=0, w_init_max=100):
        self.layers=number_of_hidden_layers
        self.output_layer_index=self.layers+1
        self.ius=number_of_input_units
        self.ous=number_of_output_units
        self.hidden_layer_shape={0:self.ius, self.output_layer_index:self.ous}
        
        self.hidden_layer_dict={} #inizializzare dzionario
        #each layer is represented with: layer_id: (weigth_matrix, w_0_vector)
        
        self.hidden_layer_net_and_output_dict={} #per la backpropagation
        #each layer is represented with: 
        
        self.eta=eta
    
    def add_hidden_layer(self, number_of_hidden_units, layer_id):
        h_shape=number_of_hidden_units
        if layer_id==self.layers:
            h_shape=self.hidden_layer_shape[self.output_layer_index]
            #contrassegno sulla "saturazione" delle matrici della MLP
            
        W_matrix= np.random.randint(low=w_init_min,high=w_init_max,size=(h_shape, self.hidden_layer_shape[layer_id-1]))/100
        w_bias= np.random.randint(low=w_init_min,high=w_init_max,size=(h_shape))/100 
        self.hidden_layer_dict[layer_id]= (W_matrix, w_bias)

    def feedfoward(self, x):
        input=x
        for i in range(self.layers):
            net= self.hidden_layer_dict[i][0].dot(input) + self.hidden_layer_dict[i][1]
            #salvare la net e l'output nell'oggetto MLP per la backpropagation
            output=sigmoid(net)
            input=output
        return output
    
    def backpropagation(self):
        pass


#NN=MLP(3,2,1,0.5)
#x_1=np.array([1,3,5])
#print(NN.feedfoward(x_1))