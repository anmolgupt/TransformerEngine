from . import fp8_cast
import torch
import transformer_engine_extensions as tex

def clip_to_fp8(x, e5m2=False):
    amax_x = torch.max(torch.abs(x)).float()
    if e5m2:
        scale_x = 57344. / amax_x
    else:
        scale_x = 448. / amax_x
    x = fp8_cast.cast_bf16_to_fp8(x, scale_x, e5m2)
    x = fp8_cast.cast_fp8_to_bf16(x, 1./scale_x, e5m2)
    return x

def set_current_amax(x, x_fp8_format, fp8_meta_tensor, tensor_type):
    #print("Calling fp8_cast_transpose_fused")
    amax_x = torch.max(torch.abs(x)).float()
    #print("---Scale and Scale inv before update: ", fp8_meta_tensor.scale[tensor_type], fp8_meta_tensor.scale_inv[tensor_type])
    #print("---AMAX VAL: ", amax_x)
    #print("---Before Amax history: ",tensor_type, fp8_meta_tensor.amax_history)
    if amax_x > 0.0 and torch.isfinite(amax_x):
        if x_fp8_format==tex.DType.kFloat8E5M2:
            fp8_meta_scale = 57344. / amax_x
        else:
            fp8_meta_scale = 448. / amax_x
        fp8_meta_scale_inv = 1. / fp8_meta_scale
        fp8_meta_tensor.scale[tensor_type].copy_(fp8_meta_scale)
        fp8_meta_tensor.scale_inv[tensor_type].copy_(fp8_meta_scale_inv)
        #print("---Scale and Scale inv after update: ", fp8_meta_tensor.scale[tensor_type], fp8_meta_tensor.scale_inv[tensor_type])
    #print("Before Amax history: ", fp8_meta_tensor.amax_history[0][tensor_type])
