# Quantizations
# Post-Training Quantization for dummy
Following papers are implemented
[Quantizing deep convolutional networks forefficient inference: A whitepaper](https://arxiv.org/abs/1806.08342) <br />
[Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721) <br />

| Bitwidth      | Reproduced result | 
|-----------|---------:|
| 8 bitwidth - Batch Norm folding |    0.1%    | 
| 8 bitwidth - Cross Layer Equalized|    69.05%    | 

## Note
BatchNorm layer has been folded<br />
Convolution layer and Activations are fused as one operation<br />
For Data Free Quantization, only Cross Lyaer Equalization has been implemented<br />

```
python main.py --images-dir <data-path> --ptq cle
```

### reference
MobileNet pretrained model : https://github.com/tonylins/pytorch-mobilenet-v2
