import torch


def quantize_tensor(x, n_bit=8, axis=0):
    assert n_bit in [2, 4, 8], "Only 2-bit, 4-bit, and 8-bit quantization are supported"
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

    # Pack low-bit tensors into 8-bit dtype
    if n_bit < 8:
        x_int8 = pack_low_bit_tensor(x_int8, n_bit)

    return x_int8, scales, zeros


def dequantize_tensor(x, scales, zeros, orig_shape, n_bit=8, axis=0):
    assert n_bit in [2, 4, 8], "Only 2-bit, 4-bit, and 8-bit quantization are supported"
    # Unpack low-bit tensor from 8-bit dtype
    if n_bit < 8:
        x = unpack_low_bit_tensor(x, n_bit, orig_shape)

    # Move the quantization axis to the first dimension
    x = x.transpose(0, axis)

    return (
        x.sub(2 ** (n_bit - 1))
        .mul(scales.reshape(-1, *([1] * (x.dim() - 1))))
        .add(zeros.reshape(-1, *([1] * (x.dim() - 1))))
        .reshape_as(x)
        .transpose(0, axis)
    )


def pack_low_bit_tensor(tensor, n_bit):
    assert n_bit in [2, 4], "Only 2-bit and 4-bit packing are supported"

    if n_bit == 4:
        assert torch.all(tensor < 16) and torch.all(
            tensor >= 0
        ), "All values must be in [0, 15] range for 4-bit packing"
    else:
        # 2-bit packing
        assert torch.all(tensor < 4) and torch.all(
            tensor >= 0
        ), "All values must be in [0, 3] range for 2-bit packing"

    values_per_byte = 8 // n_bit

    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Pad the tensor if necessary
    if flat_tensor.numel() % values_per_byte != 0:
        padding_size = values_per_byte - (flat_tensor.numel() % values_per_byte)
        flat_tensor = torch.cat([flat_tensor, flat_tensor.new_zeros(padding_size)])

    # Reshape to 2D tensor
    reshaped = flat_tensor.reshape(-1, values_per_byte)

    shifts = torch.arange(0, 8, n_bit, device=tensor.device)
    packed = (reshaped << shifts).sum(dim=1).byte()

    return packed


def unpack_low_bit_tensor(packed_tensor, n_bit, original_shape):
    assert n_bit in [2, 4], "Only 2-bit and 4-bit unpacking are supported"

    mask = (1 << n_bit) - 1

    # Calculate the total number of elements in the original tensor
    original_numel = torch.prod(torch.tensor(original_shape))

    shifts = torch.arange(0, 8, n_bit, device=packed_tensor.device)
    unpacked = ((packed_tensor.unsqueeze(1) >> shifts) & mask).flatten()

    # Flatten and truncate to the original number of elements
    original = unpacked.reshape(-1)[:original_numel]

    # Reshape back to original shape
    original = original.reshape(original_shape)

    return original
