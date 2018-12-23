import pytest
import torch


@pytest.fixture(scope='module')
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
