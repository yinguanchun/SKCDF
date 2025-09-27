## A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation (CVPR 2025)
Official code for "A Semantic Knowledge Complementarity based Decoupling Framework
for Semi-supervised Class-imbalanced Medical Image Segmentation". (CVPR 2025)
# Training

When training on the Synapse dataset, the hyperparameters are as follows:
```
max_epoch=1500, cps_loss='w_ce+dice', sup_loss='w_ce+dice', batch_size=2, num_workers=2, base_lr=0.3, ema_w=0.99, cps_w=10, cps_rampup=True, consistency_rampup=None
```
When training on the AMOS dataset, the hyperparameters are as follows:
```
max_epoch=1500, cps_loss='w_ce+dice', sup_loss='w_ce+dice', batch_size=2, num_workers=2, base_lr=0.1, ema_w=0.99, cps_w=10, cps_rampup=True, consistency_rampup=None
```
Then run ```train_skcdf.py``` to train.
# Testing
Run ```test.py``` to generate prediction results.
# Evaluating
Run ```evaluate_Ntimes```  to calculate average Dice and ASD.


# Citation
If this code is useful for your research, please cite:
```bibtex
@inproceedings{zhang2025semantic,
  title={A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation},
  author={Zhang, Zheng and Yin, Guanchun and Zhang, Bo and Liu, Wu and Zhou, Xiuzhuang and Wang, Wendong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25940--25949},
  year={2025}
}
```
# Acknowledgements
This project is built upon the following outstanding open-source projects: [GenericSSL](https://github.com/xmed-lab/GenericSSL), [AllSpark](https://github.com/xmed-lab/AllSpark) and [ABC](https://github.com/LeeHyuck/ABC). We deeply appreciate their contributions to the community.
