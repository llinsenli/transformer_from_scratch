from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": 'en',
        "lang_tgt": 'it',
        "model_folder": 'weights',
        "model_basename": 'tmodel_',
        "preload": f'04', # The latest store model in which epoch
        "tokenizer_file": 'tokenizer_{0}.json',
        "experiemnt_name": 'run/tmodel',
        "subsample": True
    }

def get_weights_file_path(config, epoch: str):
    '''
    Get the path of the store model indicate by epoch
    '''
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)


