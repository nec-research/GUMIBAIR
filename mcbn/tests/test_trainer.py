from mcbn.cmvib import CMVIB

def test_train(state):
    """
    Take the state fixture as an input.
    Check if state contains all expected keys.
    Check if any of the values of state are None.
    """
    state_keys = [
        'state_dict', 
        'best_val_score', 
        'n_latents',
        'abundance_dim',
        'marker_dim', 
        'device', 
        'optimizer', 
        'epoch', 
        'cond_dim',
        'hidden_abundance',
        'hidden_markers',
        'hidden_decoder', 
        'n_heads', 
        'p_drop', 
        'out_dim',
        'run_mode',
        ]
        
    assert list(state.keys()) == state_keys, "Keys of state don't match the expected!"
    assert any(value is None for value in list(state.values())) == False, "Some value of state is None!"

def test_evaluate(trainer, train_test_split):
    """
    Test Trainer.evaluate() and check if output matches the expectation
    """
    _, val_loader = train_test_split[0]

    evaluation_output = trainer.evaluate(val_loader, epoch=0)

    assert (type(evaluation_output) == tuple) & (len(evaluation_output) == 3)

def test_load_checkpoint(trainer, state):
    """
    Test Trainer.load_checkpoint() and check the output.
    """
    model, _ = trainer.load_checkpoint(state)
    assert isinstance(model, CMVIB)

    