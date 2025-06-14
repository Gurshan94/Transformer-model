from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, greedy_decode
from tokenizers import Tokenizer
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)
config = get_config()

parser = argparse.ArgumentParser()
parser.add_argument("--sentence", type=str, required=True, help="English sentence to translate")
args = parser.parse_args()

tokenizer_src_path = Path(config ['tokenizer_file'].format('src'))
tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))

tokenizer_tgt_path = Path(config ['tokenizer_file'].format('tgt'))
tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

enc_input_tokens = tokenizer_src.encode(args.sentence).ids

enc_num_padding_tokens = config['seq_len'] - len(enc_input_tokens) - 2

if enc_num_padding_tokens < 0:
    raise ValueError('Sentence is too long')

sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

encoder_input = torch.cat(
    [
        sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        eos_token,
        torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
    ]
)

encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()


model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
model_filename = get_weights_file_path(config, f"00")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

model_out = greedy_decode(
    model,
    encoder_input.unsqueeze(0).to(device),  # Add batch dimension
    encoder_mask.to(device),
    tokenizer_src,
    tokenizer_tgt,
    config['seq_len'],
    device
)

model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
print(f'Source: {args.sentence}')
print(f'Translation: {model_out_text}')