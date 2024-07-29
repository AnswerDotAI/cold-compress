import torch


def quantize_tensor(x, n_bit=8, axis=0):
    # Move the quantization axis to the first dimension
    x = x.transpose(0, axis)

    min_val, max_val = torch.aminmax(x.reshape(x.shape[0], -1), dim=1)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))

    x_int8 = (
        x.sub(min_val.reshape(-1, *([1] * (x.dim() - 1))))
        .div(scales.reshape(-1, *([1] * (x.dim() - 1))))
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int8)
        .reshape_as(x)
    ).transpose(0, axis)

    return x_int8, scales, zeros


def dequantize_tensor(x, scales, zeros, n_bit=8, axis=0):
    # Move the quantization axis to the first dimension
    x = x.transpose(0, axis)

    return (
        x.sub(2 ** (n_bit - 1))
        .mul(scales.reshape(-1, *([1] * (x.dim() - 1))))
        .add(zeros.reshape(-1, *([1] * (x.dim() - 1))))
        .reshape_as(x)
        .transpose(0, axis)
    )
