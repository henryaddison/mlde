"""Helper methods"""


def param_count(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size(model):
    """Compute size in memory of model in MB."""
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )

    return (param_size + buffer_size) / 1024**2
