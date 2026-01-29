"""Unit tests for ZipMamba model components."""

import pytest
import torch

from zipmamba.model.conv_embed import ConvEmbed
from zipmamba.model.sampling import Downsample, Upsample, Bypass
from zipmamba.model.conv_module import ConvModule
from zipmamba.model.feedforward import FeedForward


# Check CUDA availability for mamba tests
CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA required for mamba-ssm"
)


class TestConvEmbed:
    def test_forward_shape(self):
        embed = ConvEmbed(input_dim=80, embed_dim=192)
        x = torch.randn(2, 100, 80)
        out = embed(x)
        assert out.shape == (2, 50, 192)  # time//2

    def test_output_length(self):
        embed = ConvEmbed(input_dim=80, embed_dim=192)
        assert embed.get_output_length(100) == 50
        assert embed.get_output_length(101) == 51


class TestSampling:
    def test_downsample(self):
        down = Downsample(factor=2)
        x = torch.randn(2, 100, 256)
        out = down(x)
        assert out.shape == (2, 50, 256)

    def test_upsample(self):
        up = Upsample(factor=2)
        x = torch.randn(2, 50, 256)
        out = up(x)
        assert out.shape == (2, 100, 256)

    def test_bypass_same_dim(self):
        bypass = Bypass(dim=256)
        x_in = torch.randn(2, 100, 256)
        x_out = torch.randn(2, 100, 256)
        out = bypass(x_in, x_out)
        assert out.shape == (2, 100, 256)


class TestMamba:
    @requires_cuda
    def test_bimamba_block(self):
        from zipmamba.model.mamba import BiMambaBlock

        block = BiMambaBlock(dim=256, d_state=16).cuda()
        x = torch.randn(2, 50, 256).cuda()
        out = block(x)
        assert out.shape == (2, 50, 256)


class TestConvModule:
    def test_forward(self):
        conv = ConvModule(dim=256, kernel_size=31)
        x = torch.randn(2, 50, 256)
        out = conv(x)
        assert out.shape == (2, 50, 256)


class TestFeedForward:
    def test_forward(self):
        ff = FeedForward(dim=256, expand=4)
        x = torch.randn(2, 50, 256)
        out = ff(x)
        assert out.shape == (2, 50, 256)


class TestConMambaBlock:
    @requires_cuda
    def test_forward(self):
        from zipmamba.model.conmamba_block import ConMambaBlock

        block = ConMambaBlock(dim=256).cuda()
        x = torch.randn(2, 50, 256).cuda()
        out = block(x)
        assert out.shape == (2, 50, 256)


class TestEncoder:
    @requires_cuda
    def test_encoder_forward(self):
        from zipmamba.model.encoder import EfficientASREncoder

        encoder = EfficientASREncoder().cuda()
        x = torch.randn(2, 800, 80).cuda()
        lengths = torch.tensor([800, 600]).cuda()

        out, out_lengths = encoder(x, lengths)

        assert out.shape[0] == 2
        assert out.shape[2] == encoder.output_dim
        assert out_lengths is not None

    @requires_cuda
    def test_model_forward(self):
        from zipmamba.model.encoder import EfficientASRModel

        model = EfficientASRModel(vocab_size=1000).cuda()
        x = torch.randn(2, 800, 80).cuda()
        lengths = torch.tensor([800, 600]).cuda()

        logits, out_lengths = model(x, lengths)

        assert logits.shape[0] == 2
        assert logits.shape[2] == 1000
        assert out_lengths is not None

    @requires_cuda
    def test_ctc_loss(self):
        from zipmamba.model.encoder import EfficientASRModel

        model = EfficientASRModel(vocab_size=1000).cuda()
        x = torch.randn(2, 800, 80).cuda()
        x_lengths = torch.tensor([800, 600]).cuda()
        targets = torch.randint(1, 100, (2, 20)).cuda()
        target_lengths = torch.tensor([20, 15]).cuda()

        loss = model.compute_loss(x, x_lengths, targets, target_lengths)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    @requires_cuda
    def test_greedy_decode(self):
        from zipmamba.model.encoder import EfficientASRModel

        model = EfficientASRModel(vocab_size=1000).cuda()
        x = torch.randn(2, 800, 80).cuda()
        lengths = torch.tensor([800, 600]).cuda()

        decoded = model.decode_greedy(x, lengths)

        assert len(decoded) == 2
        assert all(isinstance(d, list) for d in decoded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
