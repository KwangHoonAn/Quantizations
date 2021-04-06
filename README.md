# Quantizations
# Post-Training Quantization for dummy
Following papers are implemented in PyTorch (DFQ)
[Quantizing deep convolutional networks forefficient inference: A whitepaper](https://arxiv.org/abs/1806.08342) <br />
[Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721) <br />

| Quantization with min-max      | Reproduced result | Paper result | 
|-----------|-----------|---------:|
| 8 bitwidth - Batch Norm folding |    0.1%    |     0.1%    | 
| 8 bitwidth - Cross Layer Scaling|    69.59%    |  69.91%    | 
| 8 bitwidth - CLS + High bias Absortion | 70.02% | 70.92% |

| Quantization with Mean Squared Error     | Reproduced result | Paper result | 
|-----------|-----------|---------:|
| 8 bitwidth - Batch Norm folding |    0.1%    |     0.11%    | 
| 8 bitwidth - Cross Layer Scaling|    70.12%    |  69.91%    | 
| 8 bitwidth - CLS + High bias Absortion | 70.36% | 70.92% |

## Note
BatchNorm layer has been folded<br />
Convolution layer and Activations are fused as one operation<br />
Cross Layer scaling & High bias fold are implemented <br />

```
python main.py --images-dir <data-path> --ptq cle hba
```
## Packages
```
torch : 1.4.0+cu100
torchvision  : 0.5.0+cu100
```
### reference
MobileNet pretrained model : https://github.com/tonylins/pytorch-mobilenet-v2
