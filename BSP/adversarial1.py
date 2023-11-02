import torch
import numpy as np
from torch.autograd import Variable


class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    @staticmethod
    def forward(ctx, input):
      #ctx.iter_num = 0
      ctx.alpha = 10
      ctx.low = 0.0
      ctx.high = 1.0
      ctx.max_iter = 10000.0
      AdversarialLayer.iter_num += 1
      ctx.save_for_backward(input)
      

      return input * 1.0
    @staticmethod
    def backward(ctx, gradOutput):
      
      input, = ctx.saved_variables
      #print(AdversarialLayer.iter_num)
      coeff = float(2.0 * (ctx.high - ctx.low) / 
      (1.0 + np.exp(-ctx.alpha * AdversarialLayer.iter_num / ctx.max_iter)) - (ctx.high - ctx.low) + ctx.low)
      return -coeff * gradOutput



