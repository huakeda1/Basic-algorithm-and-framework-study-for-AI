## Lstm inputs and outputs study
Build a simple program to get a clearly understanding about the inputs and outputs of **multi-layer bidirectional** LSTM. 

### Packages
- torch

### Important functions
- torch.randn()
- torch.nn.LSTM()
- torch.eq()
- torch.cat()
- torch.view()

### Main Content

The dimensions of inputs and outputs of LSTM are depicted as below:  
output,(h_n,c_n)=model(input,(h_0,c_0))  

inputs=(input,(h_0,c_0))  
input.shape=(batch_size,seq_len,input_size)  
h_0.shape=(num_layers*directions,batch_size,hidden_size)  
c_0.shape=(num_layers*directions,batch_size,hidden_size)  

outputs=(output,(h_n,c_n))  
output.shape=(batch_size,seq_len,2*hidden_size(when bidirectional=True))  
h_n.shape=(num_layers*directions,batch_size,hidden_size)  
c_n.shape=(num_layers*directions,batch_size,hidden_size)  

h_n[-1,batch_size,hidden_size] means the last backward hidden state of the last layer(when bidirectional=True)  
h_n[-2,batch_size,hidden_size] means the last forward hidden state of the last layer(when bidirectional=True)  

h_n[-3,batch_size,hidden_size] means the last backward hidden state of the second last layer(when bidirectional=True)  
h_n[-4,batch_size,hidden_size] means the last forward hidden state of the second last layer(when bidirectional=True)  

output[:,0,hidden_size:]=h_n[-1,batch_size,hidden_size]  
output[:,-1,:hidden_size]=h_n[-2,batch_size,hidden_size]  

You can also review the [link](https://blog.csdn.net/qq_39777550/article/details/106659150) for further study.  