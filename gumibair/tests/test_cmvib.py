import torch
from gumibair.cmvib import CMVIB

def test_instantiate_cmvib(model):
    assert isinstance(model, CMVIB)

def test_modality(config, model):
    assert model.run_mode == config['mode']

def test_forward(model_input, model):
    """
    Test CMVIB.forward() behavior.
    Checks wether dimensions of the output tensors are as expected.
    """
    dataset, X, _ = model_input
    abundance, markers, __,  conditions, heads = X

    mu, logvar, classification_logits = model.forward((abundance, markers, heads, conditions))

    assert ((mu.shape == torch.Size([len(dataset), model.n_latents])) &\
            (logvar.shape == torch.Size([len(dataset), model.n_latents])))
    
    assert (classification_logits.shape == torch.Size([len(dataset), 1]))


def test_infer(model_input, model):
    """
    Test CMVIB.infer() behavior.
    Checks wether dimensions of the output tensors are as expected.
    """
    dataset, X, _ = model_input

    abundance, markers, __,  conditions, ___ = X

    infer_output = model.infer(
        abundance=abundance,
        markers=markers, 
        X_cond=conditions
    )

    assert (isinstance(infer_output, tuple)) & (len(infer_output) == 2)
    for output in infer_output:
        assert (output.shape == torch.Size([len(dataset), model.n_latents]))


def test_classify(model_input, model):
    """
    Test CMVIB.classify() behavior.
    Checks wether prediction tensor has the expected dimensions
    """
    dataset, X, device = model_input
    abundance, markers, __,  conditions, heads = X

    prediction = model.classify((abundance, markers, heads, conditions))
    
    assert ((type(prediction) == torch.Tensor) &\
            (prediction.shape == torch.Size([len(dataset), 1])))