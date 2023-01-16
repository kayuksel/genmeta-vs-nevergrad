# GenMeta-vs-Nevergrad
Comparison of Generative Meta-Learning vs Nevergrad  
on the 30-dimensional Schwefel function optimization

The best results of both methods after 100K trials:  
gen-meta best_epoch: 99500 loss: **1.597656** time: 1.372849  
ng-opt-4 best_epoch: 67590 loss: 476.789062 time: 63.584929 

average gen-meta loss after 10 repetitions: 233.1332031  
average ng-opt-4 loss after 10 repetitions: 409.2259765

Please note that, an experiment with several random  
seeds is required to correctly compare both of them.

# GenMeta in RL

**gen_meta_100k.py** Solving math functions in high-dimensions  
**gen_meta_rl.py** Training RL agents with Gen-Meta optimization  
Best score achieved in Pendulum-v1 control problem: 131.882050
