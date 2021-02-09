import torch
from torch import nn

class Quantizers(nn.Module):
	def __init__(self, bw, act_q = True, quantize = False):
		super(Quantizers, self).__init__()
		self.is_quantize = quantize
		self.act_q = act_q
		self.init = False
		self.is_symmetric = False

		self.calibration = False
		self.max_range = torch.zeros(1)[0].cuda()
		self.n = bw
		self.offset = None
		self.min = None
		self.max = None
		self.scale = 0.0

	def set_quantize(self, flag):
		self.is_quantize = flag

	def estimate_range(self, flag):
		self.calibration = flag

	def init_params(self, x_f):
		'''
		https://heartbeat.fritz.ai/quantization-arithmetic-421e66afd842
		
		There exist two modes
		1) Symmetric:
			Symmetric quantization uses absolute max value as its min/max meaning symmetric with respect to zero
		2) Asymmetric
			Asymmetric Quantization uses actual min/max, meaning it is asymmetric with respect to zero 
		
		Scale factor uses full range [-2**n / 2, 2**n - 1]
		'''

		if self.is_symmetric:
			x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
		else:
			x_min, x_max = torch.min(x_f), torch.max(x_f)

		self.max_range = torch.max(self.max_range, (x_max - x_min))
		# self.scale += (x_max - x_min) / float(2**self.n - 1)
		self.scale = self.max_range / float(2**self.n - 1)
		if not self.is_symmetric:
			self.offset = torch.round(-x_min / self.scale)

		self.init = True

	def quantize(self, x_f):
		'''
		Quantizing
		Formula is derived from below:
		https://medium.com/ai-innovation/quantization-on-pytorch-59dea10851e1
		'''
		x_int = torch.round( x_f  / self.scale )
		if not self.is_symmetric:
			x_int += self.offset

		if self.is_symmetric:
			l_bound, u_bound = -2**(self.n - 1), 2**(self.n-1) - 1
		else:
			l_bound, u_bound = 0, 2**(self.n) - 1

		x_q = torch.clamp(x_int, min = l_bound, max = u_bound)
		return x_q

	def dequantize(self, x_q):
		'''
		De-quantizing
		'''
		if not self.is_symmetric:
			x_q = x_q - self.offset
		x_dequant = x_q * self.scale
		return x_dequant

	def quant_dequant(self, x_f):
		if not self.init:
			self.init_params(x_f)
		x_q = self.quantize(x_f)
		x_dq = self.dequantize(x_q)
		return x_dq

	def forward(self, x):
		if self.calibration and self.act_q:
			self.init_params(x)
		return self.quant_dequant(x) if self.is_quantize else x



class QuantConv(nn.Module):
	def __init__(self, conv, bw = 8):
		super(QuantConv, self).__init__()
		self.conv = conv
		self.weight_quantizer = Quantizers(bw, act_q = False)
		self.kwarg = {	'stride' : self.conv.stride, \
						'padding' : self.conv.padding, \
						'dilation' : self.conv.dilation, \
						'groups': self.conv.groups}
		self.act_quantizer = Quantizers(bw)

	def batchnorm_folding(self):
		'''
		https://towardsdatascience.com/speed-up-inference-with-batch-normalization-folding-8a45a83a89d8

		W_fold = gamma * W / sqrt(var + eps)
		b_fold = (gamma * ( bias - mu ) / sqrt(var + eps)) + beta
		'''
		if hasattr(self.conv, 'gamma'):
			gamma = getattr(self.conv, 'gamma')
			beta = getattr(self.conv, 'beta')
			mu = getattr(self.conv, 'mu')
			var = getattr(self.conv, 'var')
			eps = getattr(self.conv, 'eps')

			denom = gamma.div(torch.sqrt(var + eps))

			if getattr(self.conv, 'bias') == None:
				self.conv.bias = torch.nn.Parameter(var.new_zeros(var.shape))
			b_fold = denom*(self.conv.bias.data - mu) + beta
			self.conv.bias.data.copy_(b_fold)

			denom = denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
			self.conv.weight.data.mul_(denom)

	def get_params(self):
		w = self.conv.weight.detach()
		if self.conv.bias != None:
			b = self.conv.bias.detach()
		else:
			b = None
		w = self.weight_quantizer(w)
		return w, b

	def forward(self, x):
		w, b = self.get_params()
		out = nn.functional.conv2d(input = x, weight = w, bias = b, **self.kwarg)
		
		out = self.act_quantizer(out)
		return out

class QuantLinear(nn.Module):
	def __init__(self, linear, bw = 8):
		super(QuantLinear, self).__init__()
		self.fc = linear
		self.weight_quantizer = Quantizers(bw, act_q = False)
		self.act_quantizer = Quantizers(bw)

	def get_params(self):
		w = self.fc.weight.detach()
		if self.fc.bias != None:
			b = self.fc.bias.detach()
		else:
			b = None
		w = self.weight_quantizer(w)
		return w, b

	def forward(self, x):
		w, b = self.get_params()
		out = nn.functional.linear(x, w, b)

		out = self.act_quantizer(out)
		return out

class QuantActivations(nn.Module):
	def __init__(self, activation, bw = 8):
		super(QuantActivations, self).__init__()
		self.activation_func = activation
		self.act_quantizer = Quantizers(bw)

	def forward(self, x):
		x = self.activation_func(x)
		x = self.act_quantizer(x)
		return x

class QuantBN(nn.Module):
	def __init__(self, bn, bw = 8):
		super(QuantBN, self).__init__()
		self.bn = bn
		self.bn_fold = False
		self.act_quantizer = Quantizers(bw)

	def batchnorm_folding(self):
		self.bn_fold = True

	def forward(self, x):
		return x
