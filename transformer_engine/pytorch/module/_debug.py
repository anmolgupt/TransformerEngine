from . import fp8_cast
import torch

def clip_to_fp8(x, e5m2=False):
    amax_x = torch.max(torch.abs(x)).float()
    if e5m2:
        scale_x = 57344. / amax_x
    else:
        scale_x = 448. / amax_x
    x = fp8_cast.cast_bf16_to_fp8(x, scale_x, e5m2)
    x = fp8_cast.cast_fp8_to_bf16(x, 1./scale_x, e5m2)
    return x