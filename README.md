# genmeta-vs-nevergrad
Comparison of Generative Meta-Learning and Nevergrad  
on the 30-dimensional Schwefel function optimization

The best results of both methods after 100K trials:  
gen-meta best_epoch: 99500 loss: **1.597656**  
ngopt-4 best_epoch: 67834 loss: 118.439453 (stuck at local minima)

Gen-Meta is also much faster in terms of actual speed,  
and scales easily to 100K+ dimensions in a desktop GPU.

I would say that it should be preferred to Nevergrad,  
in optimizations like NN training for non-convex RL.
