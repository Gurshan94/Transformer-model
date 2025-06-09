import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import build_transformer
from dataset import BilingualDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from config import get_config,get_weights_file_path
import warnings
from tqdm import tqdm

def get_all_sentences(ds, lang):
  for item in ds:
    yield item[lang]

def build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config ['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_ds(config):
  ds_raw = load_dataset('ai4bharat/samanantar', 'pa')

  ds_merged = [{'src':src, 'tgt':tgt} for src,tgt in zip(ds_raw['train']['src'][:100000], ds_raw['train']['tgt'][:100000])]
  del ds_raw

  tokenizer_src = build_tokenizer(config, ds_merged, 'src')
  tokenizer_tgt = build_tokenizer(config, ds_merged, 'tgt')

  train_ds_size = int(0.9 * len(ds_merged))
  val_ds_size = len(ds_merged) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_merged, [train_ds_size, val_ds_size])

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 'src', 'tgt', config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 'src', 'tgt', config['seq_len'])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_merged:
    src_ids = tokenizer_src.encode(item['src']).ids
    tgt_ids = tokenizer_tgt.encode(item['tgt']).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
  model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
  return model

def train_model(config):
  #define the device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

  initial_epoch = 0
  global_step = 0

  if config['preload'] is not None:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f'Loading model from {model_filename}')
    state = torch.load(model_filename)
    initial_epoch = state['epoch']
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
  
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

  for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')

    for batch in batch_iterator:
      encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (B, seq_len)     
      encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

      # Run the tensor through the model
      encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
      projection_output = model.project(decoder_output) # (B, seq_len, vocab_size)

      label = batch['label'].to(device) # (B, seq_len)

      # (B, seq_len, vocab_size) -> (B * seq_len, vocab_size)
      loss = loss_fn(projection_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

      writer.add_scalar('train/loss', loss.item(), global_step)
      writer.flush()

      # Backpropagation
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1

    # Save the model after each epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  config = get_config()
  train_model(config)

