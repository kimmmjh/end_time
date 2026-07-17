import pytest
import torch

from models import Decoder, RecurrentEND2D
from models.auxiliary_components import ConvGRUCell
from models.pooling_layers import TranslationalEquivariantPooling2D


def test_conv_gru_cell_is_translation_equivariant():
    torch.manual_seed(1)
    cell = ConvGRUCell(input_channels=3, hidden_channels=5, kernel_size=3)
    cell.eval()
    x = torch.randn(2, 3, 4, 4)
    hidden = torch.randn(2, 5, 4, 4)
    shift = (1, -2)

    expected = torch.roll(cell(x, hidden), shifts=shift, dims=(2, 3))
    actual = cell(
        torch.roll(x, shifts=shift, dims=(2, 3)),
        torch.roll(hidden, shifts=shift, dims=(2, 3)),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rounds", [1, 4])
def test_recurrent_decoder_accepts_variable_round_counts(rounds):
    lattice_size = 3
    decoder = Decoder(
        network=RecurrentEND2D(
            channels=[6, 8],
            depths=[1, 1],
            lattice_size=lattice_size,
            gru_channels=7,
            gru_layers=2,
        ),
        pooling=TranslationalEquivariantPooling2D(lattice_size),
        ensemble=None,
    )
    syndrome = torch.randint(
        0,
        2,
        (2, 2, rounds, lattice_size**2),
        dtype=torch.float32,
    )

    output = decoder(syndrome)

    assert output.shape == (2, 16)
    assert torch.all(torch.isfinite(output))


def test_recurrent_decoder_backpropagates_to_early_rounds():
    lattice_size = 3
    network = RecurrentEND2D(
        channels=[5],
        depths=[1],
        lattice_size=lattice_size,
        gru_channels=5,
    )
    syndrome = torch.randn(
        2, 2, 4, lattice_size**2, requires_grad=True
    )

    network(syndrome).square().mean().backward()

    assert syndrome.grad is not None
    assert torch.all(torch.isfinite(syndrome.grad))
    assert syndrome.grad[:, :, 0].abs().sum() > 0


def test_recurrent_decoder_rejects_wrong_spatial_shape():
    network = RecurrentEND2D(
        channels=[4],
        depths=[1],
        lattice_size=3,
    )

    with pytest.raises(ValueError, match="Expected syndrome shape"):
        network(torch.zeros(2, 2, 3, 8))
