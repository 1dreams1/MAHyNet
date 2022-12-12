# MAHyNet: Prediction of RNA-protein binding sites by hybrid neural network based on multi-head attention mechanism
**Introduction**
****
  In this work, we propose a new parallel deep learning model, MAHyNet, which exploits the physicochemical properties of RNA bases.  In addition, MAHyNet introduces a multi-head attention mechanism and uses a combination of convolutional neural networks and gated recurrent neural network.  MAHyNet is a parallel network.  The left branch network is a hybrid convolutional and gated cyclic neural network based on the multi-head attention mechanism.  The right branch network is a two-layer convolutional neural network based on the multi-head attention mechanism, which can extract one-hot and physicochemical properties of bases.
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
**Non-10-fold cross-validation**
****
>python generate_hdf5.py
>python generate_hdf5_ph.py
>python train_data.py 0 0 53  
>python save_result.py
****
**10-fold cross-validation**
****
>python generate_hdf5_10.py 
>python generate_hdf5_10ph.py
>python train_ten_data.py 0 0 53  
>python see_10_fold.py  
