# DATFuse
DATFuse is under minor revision. The code will be released once accepted.


## Comparison with SOTA methods

### Fusion results on TNO dataset
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/fig3.jpg)

### Fusion results on RoadScene dataset
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/fig4.jpg)


## Impact of weight parameters in the loss function

### Impact of weight parameter α on fusion performance with λ and γ ﬁxed as 100 and 10, respectively. 
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/alpha.jpg)

### Impact of weight parameter λ on fusion performance with α and γ ﬁxed as 1 and 10, respectively.
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/lambda.jpg)

### Impact of weight parameter γ on fusion performance with α and λ ﬁxed as 1 and 100, respectively.
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/gamma.jpg)

## Ablation study on network structure
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/fig6.jpg)


## Ablation study on the number of TRMs
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/ablationTRM.jpg)

## Ablation study on the second DARM
![Image text](https://github.com/tthinking/DATFuse/blob/main/imgs/woSecondDARM.jpg)


## Computational efficiency comparisons

### Average running time for generating a fused image (Unit: seconds)



| Method | TNO Dataset | RoadScene Dataset |
| :---: | :---: | :---: |
| MDLatLRR | 26.0727 | 11.7310 |
|AUIF|	0.1119 |	0.0726 |
|DenseFuse|	0.5663 |	0.3190 |
|FusionGAN|		2.6796 |	1.1442 |
|GANMcC|	5.6752 |	2.3813 |
|RFN_Nest|	2.3096|	0.9423 |
|CSF|	10.3311 |5.5395 |
|MFEIF|	0.0793 	|0.0494 |
|PPTFusion|		1.4150 |0.8656 |
|SwinFuse|	3.2687 |	1.6478 |
|DATFuse|	0.0257 |	0.0141|
