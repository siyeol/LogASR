import torch

from log_asr.quantization import log_a4_fake_quant, uniform_w4_fake_quant


def test_quantizers_preserve_shape_dtype_and_zero():
    x = torch.tensor([[0.0, -1.0, 0.25, 2.0, -3.0]], dtype=torch.float16)
    outputs = (uniform_w4_fake_quant(x, group_size=4), log_a4_fake_quant(x))
    for output in outputs:
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output[0, 0].item() == 0.0
        assert torch.isfinite(output).all()


def test_dynamic_log_maps_group_absmax_exactly():
    x = torch.tensor([[0.0, 1.0, -2.0, 4.0]], dtype=torch.float32)
    output = log_a4_fake_quant(x)
    assert torch.isclose(output.abs().max(), x.abs().max())



def test_log_scale_is_per_token_and_grouping_free():
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0], [0.5, 1.0, 2.0, 4.0]]])
    output = log_a4_fake_quant(x)
    assert torch.allclose(output.abs().amax(dim=-1), x.abs().amax(dim=-1))
