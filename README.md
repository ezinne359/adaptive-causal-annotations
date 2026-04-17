# Batch Adaptive Causal Annotations
Code for [*Batch Adaptive Causal Annotations*](https://arxiv.org/pdf/2502.10605) by Nwankwo, Goldkind, and Zhou (2026). To appear in AISTATS. 

The repository contains notebooks with examples:
- simulated data ([```experiments_simulated.ipynb```](https://github.com/ezinne359/adaptive-causal-annotations/blob/main/experiments_simulated.ipynb))
- retail hero data ([```experiments_retailhero.ipynb```](https://github.com/ezinne359/adaptive-causal-annotations/blob/main/experiments_retailhero.ipynb))
- street outreach data (```omitted due to privacy concerns```)

## Additional Notes 

- utils.py contains the batch adaptive causal annotation algorithm and aipw estimation procedure
- forestriesz.py was adapted for the balancing estimator from [Chernozhukov, Newey, Quintas-Martínez and Syrgkanis (2021) "RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests"](https://github.com/victor5as/RieszLearning)
- plotting_paper_results.ipynb plots simulated and retailhero experimental results in the paper. Results are stored in data file or can be recreated from the notebooks above
- data folder contains csv files with results dataset used in ```plotting_paper_results.ipynb``` and agumented retailhero dataset with offline LLM predictions

