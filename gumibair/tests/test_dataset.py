from gumibair.dataset import FullMicrobiomeDataset
import pandas as pd
import torch
import numpy as np
import pytest

@pytest.fixture
def mock_fmd(mocker, config):
    mock_fmd =  mocker.MagicMock(spec=FullMicrobiomeDataset, autospec=True)
    mock_fmd.cohorts = ["cohort1"]
    mock_fmd.data_dir = "data_dir"
    mock_fmd.data = "joint"
    mock_fmd.device = config['device']
    mock_fmd.start_idx_df = 1
    mock_fmd.drop_samples = None
    mock_fmd.conditioning_labels = None

    return mock_fmd

@pytest.fixture
def mock_data():
    input_dict = {
        "Unnamed: 0": ['column1', 'column2'],
        "sample1": ["A", 0],
        "sample2": ["B", 5],
        "sample3": ["B", 10]
    }
    mock_data = pd.DataFrame().from_dict(input_dict)
    mock_data.index = ['column1', 'column2']
    
    return mock_data
    
def test_load_disease_data(config):
    #TODO mock out the actual data import
    function = FullMicrobiomeDataset.load_disease_data
    cohort = list(config['cohorts_to_disease'].keys())[0]
    data_dir = config['path']
    raw_data = function(cohort, data_dir, data="joint")

    assert isinstance(raw_data, dict) & (len(raw_data) == 2)
    assert all(isinstance(value, pd.DataFrame) for value in raw_data.values())
 
def test_load_disease_data_drop_samples(config):
    #TODO mock out the actual data import
    function = FullMicrobiomeDataset.load_disease_data
    cohort = list(config['cohorts_to_disease'].keys())[0]
    data_dir = config['path']

    call_1_output = function(cohort, data_dir, data="joint")['marker']
    sample =call_1_output.columns[0]

    drop_samples={cohort:sample}
    call_2_output = function(cohort, data_dir, data="joint", drop_samples=drop_samples)['marker']
    assert call_2_output.shape[1] == call_1_output.shape[1]-1
    assert sample not in call_2_output.columns

def test_get_labels(mocker, mock_fmd):
    mock_fmd._get_labels = mocker.MagicMock(wraps=FullMicrobiomeDataset._get_labels)
    raw_data={
        "marker":pd.DataFrame(dict(
            sample1=["CRC"],
            sample2=["healthy"]
            ),
            index=["disease"]
        )
    }
    labels = mock_fmd._get_labels(mock_fmd, raw_data, "Colorectal")
    assert isinstance(labels, torch.Tensor) & (labels.shape == torch.Size([2,1]))

def test_get_metadata(mocker, mock_fmd, mock_data):
    mock_fmd._get_metadata = mocker.MagicMock(wraps=FullMicrobiomeDataset._get_metadata)
    mocker.patch('pandas.read_csv', return_value=mock_data)
    metadata = mock_fmd._get_metadata(mock_fmd)
    assert (metadata['column1'] == ['A', 'B', 'B']) & (metadata['column2'] == [0,5,10])

def test_get_metadata_drop_samples(mock_fmd, mocker, mock_data):
    mock_fmd._get_metadata = mocker.MagicMock(wraps=FullMicrobiomeDataset._get_metadata)
    mock_fmd.drop_samples = {"cohort1":["sample1"]}
    mocker.patch('pandas.read_csv', return_value=mock_data)
    metadata = mock_fmd._get_metadata(mock_fmd)
    assert (metadata['column1'] == ['B','B']) & (metadata['column2'] == [5,10])

def test_get_metadata_conditioning_labels(mocker, mock_fmd, mock_data):
    mock_fmd._get_metadata = mocker.MagicMock(wraps=FullMicrobiomeDataset._get_metadata)
    mock_fmd.conditioning_type="multi"
    mock_fmd.conditioning_labels = {'column1': 'categorical', 'column2': 'numerical'}
    mocker.patch('pandas.read_csv', return_value=mock_data)
    metadata = mock_fmd._get_metadata(mock_fmd)
    assert (metadata['column1'] == ['A', 'B']) & (metadata['column2'] == [0,5,10])

def test_encode_conditions(mocker, mock_fmd, mock_data):
    mock_fmd._encode_conditions = mocker.MagicMock(wraps=FullMicrobiomeDataset._encode_conditions)
    mock_fmd.conditioning_type="multi"
    mock_fmd.conditioning_labels = {'column1': 'categorical', 'column2': 'numerical'}
    mock_fmd.metadata = {
        'column1': ['A', 'B'],
        'column2': [0, 5, 10]
    }
    mock_data.drop(columns=['Unnamed: 0'], inplace=True)
    mock_raw_data = {'marker': mock_data}
    labels = [0,1,0]
    conditions = mock_fmd._encode_conditions(mock_fmd, mock_raw_data, labels)
    assert (isinstance(conditions, np.ndarray)) & (conditions.shape == (3,3))
    assert (conditions[:, -1] == np.array([0., 0.5, 1.], dtype=float)).all()
