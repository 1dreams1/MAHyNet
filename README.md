# MAHyNet: Prediction of RNA-protein binding sites by hybrid neural network based on multi-head attention mechanism
**Introduction**
****
  In this study, we propose a new deep-learning-based model, named MAHyNet, which combines a multi-head attention mechanism, a convolutional neural network, and a gated recurrent neural network. Specifically, the multi-head attention mechanism is a collection of multiple independent attention layers, which can extract sequence feature information from multiple dimensions. The combination of convolutional neural network and gated recurrent neural network can further extract high-level features of the sequence. Furthermore, we explored the effect of hyper-parameters on the model performance, and used global pooling and local pooling for down-sampling in the model, simplifying model complexity and improving the model performance, resulting in better prediction of RNA-protein binding sites. 
****
**Requirements**
****
* Keras = 2.1.6  
* tensorflow-gpu =1.8.0  
* h5py  
* pool  
* tqdm  
* sklearn
****
**10-fold cross-validation**
****
>python generate_hdf5_10.py  
>python train_ten_data.py 0 0 53  
>python see_10_fold.py  
