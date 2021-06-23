# Errata
In the notebook covering [batch normalization](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter03/Batch_normalization.ipynb), the second half of the experiment was missing a `model.eval()` in `val_loss` function. 
Due to which the graphs were not accurate. 

The graph presented in page 147 shown below  
![image](https://user-images.githubusercontent.com/3656100/123093860-ceabd180-d449-11eb-94b5-27b606b4d65d.png)
  
should actually be  
![image](https://user-images.githubusercontent.com/3656100/123092602-4bd64700-d448-11eb-9244-27b0e7402393.png)
