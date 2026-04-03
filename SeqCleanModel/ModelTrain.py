import os
import glob
import json
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, BertConfig, BertModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import Counter
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np
import random
import signal
import sys
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    logger.info("Received termination signal, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def build_custom_tokenizer(data, vocab_file='vocab.json'):
    if os.path.exists(vocab_file):
        logger.info(f"Found existing vocabulary file {vocab_file}, attempting to load")
        try:
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            logger.info(f"Successfully loaded vocabulary, size: {len(vocab)}")
            return vocab
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}, rebuilding")

    logger.info("Building vocabulary from scratch")
    word_counts = Counter()
    for d in data:
        word_counts.update(d['input'].split())
        word_counts.update(d['target'].split())
    vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
    vocab.update({word: i+4 for i, (word, _) in enumerate(word_counts.most_common())})
    logger.info(f"Built vocabulary, size: {len(vocab)}")

    try:
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f)
        logger.info(f"Vocabulary saved to {vocab_file}")
    except Exception as e:
        logger.warning(f"Failed to save vocabulary: {e}")

    return vocab


def tokenize_sequence(seq, vocab, max_length):
    tokens = seq.split()
    token_ids = [vocab.get(token, vocab['[UNK]']) for token in tokens]
    token_ids = [vocab['[CLS]']] + token_ids[:max_length-2] + [vocab['[SEP]']]
    padding = [vocab['[PAD]']] * (max_length - len(token_ids))
    attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
    return token_ids + padding, attention_mask


def load_data(data_path, folder_list):
    data = []
    folder_stats = {}

    def read_file(file_path):
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        for folder in folder_list:
            folder_name = folder.split('/')[-1]
            clean_file = os.path.join(folder, 'clean_date.txt')
            if not os.path.exists(clean_file):
                logger.warning(f"clean_date.txt not found in {folder_name}")
                folder_stats[folder_name] = 0
                continue
            clean_seq = executor.submit(read_file, clean_file).result()
            if not clean_seq:
                logger.warning(f"clean_date.txt is empty in {folder_name}")
                folder_stats[folder_name] = 0
                continue
            fixed_files = [os.path.join(folder, f'fixed_date_{i}.txt') for i in range(1, 101)
                           if os.path.exists(os.path.join(folder, f'fixed_date_{i}.txt'))]
            results = executor.map(read_file, fixed_files)
            valid_files = 0
            for fixed_file, fixed_seq in zip(fixed_files, results):
                if fixed_seq:
                    data.append({'input': fixed_seq, 'target': clean_seq})
                    valid_files += 1
                else:
                    logger.warning(f"{fixed_file} is empty or invalid")
            folder_stats[folder_name] = valid_files

    logger.info(f"Loaded {len(data)} data pairs from {len(folder_list)} folders")
    logger.info(f"Folder stats (first 10): { {k: v for k, v in list(folder_stats.items())[:10] if v > 0} }")
    return data


def split_folders(data_path, start_folder, end_folder):
    folders = [f for f in glob.glob(os.path.join(data_path, '*'))
               if os.path.isdir(f) and f.split('/')[-1].isdigit()
               and int(start_folder) <= int(f.split('/')[-1]) <= int(end_folder)]
    folders.sort(key=lambda x: int(x.split('/')[-1]))
    total_folders = len(folders)
    if total_folders == 0:
        raise ValueError("No valid folders found")

    val_size = max(1, int(total_folders * 0.1))
    test_size = max(1, int(total_folders * 0.1))
    val_folders = folders[:val_size]
    test_folders = folders[val_size:val_size + test_size]
    train_folders = folders[val_size + test_size:]

    logger.info(f"Total folders: {total_folders}")
    logger.info(f"Train folders: {len(train_folders)} ({train_folders[0].split('/')[-1]} - {train_folders[-1].split('/')[-1]})")
    logger.info(f"Val   folders: {len(val_folders)} ({val_folders[0].split('/')[-1]} - {val_folders[-1].split('/')[-1]})")
    logger.info(f"Test  folders: {len(test_folders)} ({test_folders[0].split('/')[-1]} - {test_folders[-1].split('/')[-1]})")
    return train_folders, val_folders, test_folders


def preprocess_data(data, vocab, max_length, dataset_name="dataset"):
    inputs = []
    targets = []
    attention_masks = []
    invalid_count = 0
    for d in data:
        if d['input'].strip() and d['target'].strip():
            input_ids, input_mask = tokenize_sequence(d['input'], vocab, max_length)
            target_ids, _ = tokenize_sequence(d['target'], vocab, max_length)
            inputs.append(input_ids)
            targets.append(target_ids)
            attention_masks.append(input_mask)
        else:
            invalid_count += 1
    logger.info(f"{dataset_name}: processed {len(inputs)} valid input-target pairs, filtered {invalid_count} invalid pairs")
    return {
        'input_ids': inputs,
        'labels': targets,
        'attention_mask': attention_masks
    }


def get_max_length(data, train_dataset=None, config_file='config.json'):
    if os.path.exists(config_file):
        logger.info(f"Found config file {config_file}, attempting to load max_length")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            max_length = config.get('max_length', None)
            if max_length is not None:
                logger.info(f"Successfully loaded max_length: {max_length}")
                return max_length
        except Exception as e:
            logger.warning(f"Failed to load max_length: {e}")

    if train_dataset is not None:
        logger.info("Inferring max_length from cached dataset")
        try:
            max_length = max(len(ids) for ids in train_dataset['input_ids'] + train_dataset['labels'])
            logger.info(f"Inferred max_length from dataset: {max_length}")
            try:
                config = {'max_length': max_length}
                with open(config_file, 'w') as f:
                    json.dump(config, f)
            except Exception:
                pass
            return max_length
        except Exception as e:
            logger.warning(f"Failed to infer max_length from dataset: {e}")

    logger.info("Computing max sequence length from raw data")
    max_input_len = max(len(d['input'].split()) for d in data)
    max_target_len = max(len(d['target'].split()) for d in data)
    # 5000 tokens with standard self-attention is very memory-intensive;
    # use fp16 and gradient_checkpointing to fit in GPU memory.
    max_length = min(max(max_input_len, max_target_len) + 2, 5000)
    logger.info(f"Computed max sequence length: {max_length}")

    try:
        config = {'max_length': max_length}
        with open(config_file, 'w') as f:
            json.dump(config, f)
        logger.info(f"max_length saved to config.json")
    except Exception as e:
        logger.warning(f"Failed to save max_length: {e}")

    return max_length


class CustomDataCollator:
    def __init__(self, pad_token_id, max_length):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CustomEncoderDecoderModel(EncoderDecoderModel):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__(encoder=encoder, decoder=decoder)
        self.lm_head = nn.Linear(decoder.config.hidden_size, vocab_size)
        self.config.vocab_size = vocab_size
        self.config.pad_token_id = 0

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(
            input_ids=labels,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
        )


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, vocab=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.eval_history = []
        self.eval_count = 0
        self.start_time = time.time()
        self.val_parts_dir = './val_parts'
        self.val_parts = []

    def split_val_dataset(self, val_dataset, num_parts=100):
        os.makedirs(self.val_parts_dir, exist_ok=True)
        val_size = len(val_dataset)
        part_size = val_size // num_parts
        indices = list(range(val_size))
        random.shuffle(indices)
        for i in range(num_parts):
            part_indices = indices[i * part_size:(i + 1) * part_size] if i < num_parts - 1 else indices[i * part_size:]
            part_dataset = val_dataset.select(part_indices)
            part_path = os.path.join(self.val_parts_dir, f'part_{i+1:03d}')
            part_dataset.save_to_disk(part_path)
            self.val_parts.append(part_path)
        logger.info(f"Validation set split into {num_parts} parts")

    def load_random_val_part(self):
        if not self.val_parts:
            self.val_parts = glob.glob(os.path.join(self.val_parts_dir, 'part_*'))
            if not self.val_parts:
                logger.error("Validation subsets not prepared, exiting")
                sys.exit(1)
        part_path = random.choice(self.val_parts)
        part_dataset = Dataset.load_from_disk(part_path)
        return part_dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels']
        }
        outputs = model(**model_inputs)
        loss = outputs.loss
        if loss is not None and loss.dim() > 0:
            loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        eval_dataset = self.load_random_val_part()
        logger.info(f"Running validation subset evaluation, samples: {len(eval_dataset)}")
        metrics = super().evaluate(eval_dataset=eval_dataset, *args, **kwargs)
        exact_match = metrics.get('eval_exact_match', 0.0)
        logger.info(f"Validation subset result: exact_match={exact_match}")

        self.eval_count += 1
        self.eval_history.append(exact_match)

        if len(self.eval_history) >= 5:
            recent_evals = self.eval_history[-5:]
            max_diff = max(recent_evals) - min(recent_evals)
            if max_diff < 0.008:
                logger.info(f"Last 5 eval scores vary by less than 0.008, triggering early stopping")
                self.control.should_training_stop = True

        return metrics

    def train(self, *args, **kwargs):
        self.start_time = time.time()
        total_steps = self.args.max_steps if self.args.max_steps > 0 else int(len(self.train_dataset) / self.args.per_device_train_batch_size * self.args.num_train_epochs)

        output = super().train(*args, **kwargs)

        elapsed_time = time.time() - self.start_time
        steps_completed = self.state.global_step
        if steps_completed > 0:
            logger.info(f"Training completed {steps_completed} steps in {timedelta(seconds=int(elapsed_time))}")
        return output


def train(data_path, start_folder, end_folder, quick_validate=False):
    try:
        train_folders, val_folders, test_folders = split_folders(data_path, start_folder, end_folder)

        train_dataset_path = 'tokenized_train_dataset'
        val_dataset_path = 'tokenized_val_dataset'
        train_dataset = None
        val_dataset = None

        if os.path.exists(train_dataset_path):
            logger.info(f"Found cached training dataset, attempting to load")
            try:
                train_dataset = Dataset.load_from_disk(train_dataset_path)
                logger.info(f"Successfully loaded training dataset, size: {len(train_dataset)}")

                if quick_validate:
                    train_size = len(train_dataset)
                    reduced_size = max(1, train_size // 10)
                    indices = random.sample(range(train_size), reduced_size)
                    train_dataset = train_dataset.select(indices)
                    logger.info(f"[Quick-validate] Training set reduced to 1/10, size: {len(train_dataset)}")
            except Exception as e:
                logger.warning(f"Failed to load training dataset: {e}, will reprocess")

        if os.path.exists(val_dataset_path):
            try:
                val_dataset = Dataset.load_from_disk(val_dataset_path)
                logger.info(f"Successfully loaded validation dataset, size: {len(val_dataset)}")
            except Exception as e:
                logger.warning(f"Failed to load validation dataset: {e}, will reprocess")

        train_data = None
        val_data = None
        if train_dataset is None or val_dataset is None:
            logger.info("Loading raw training and validation data")
            train_data = load_data(data_path, train_folders)
            val_data = load_data(data_path, val_folders)

            if not train_data or not val_data:
                logger.error("Data loading failed, exiting")
                return

            if train_data and quick_validate:
                train_size = len(train_data)
                reduced_size = max(1, train_size // 10)
                train_data = random.sample(train_data, reduced_size)
                logger.info(f"[Quick-validate] Raw training data reduced to 1/10, size: {len(train_data)}")

        vocab = None
        if train_dataset is None or val_dataset is None:
            vocab = build_custom_tokenizer(train_data + val_data)
        else:
            vocab_file = 'vocab.json'
            if os.path.exists(vocab_file):
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
            else:
                logger.error("Vocabulary file not found and dataset is cached, cannot continue")
                return

        max_length = get_max_length(train_data + val_data if train_data else [], train_dataset)

        if train_dataset is None:
            train_dict = preprocess_data(train_data, vocab, max_length, "train")
            train_dataset = Dataset.from_dict(train_dict)
            train_dataset.save_to_disk(train_dataset_path)
            del train_data

        if val_dataset is None:
            val_dict = preprocess_data(val_data, vocab, max_length, "val")
            val_dataset = Dataset.from_dict(val_dict)
            val_dataset.save_to_disk(val_dataset_path)
            del val_data

        encoder_config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=5000
        )
        decoder_config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            is_decoder=True,
            add_cross_attention=True,
            max_position_embeddings=5000
        )
        encoder = BertModel(encoder_config)
        decoder = BertModel(decoder_config)
        model = CustomEncoderDecoderModel(encoder=encoder, decoder=decoder, vocab_size=len(vocab))
        model.config.decoder_start_token_id = vocab['[CLS]']
        model.config.pad_token_id = vocab['[PAD]']
        model.config.eos_token_id = vocab['[SEP]']

        training_args = Seq2SeqTrainingArguments(
            output_dir='./results_seq2seq',
            eval_strategy='steps',
            eval_steps=2000 if not quick_validate else 100,
            save_strategy='steps',
            save_steps=2000 if not quick_validate else 100,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3 if quick_validate else 5,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=False,
            max_grad_norm=0.1,
            max_steps=5 if quick_validate else -1,
            remove_unused_columns=False,
        )

        data_collator = CustomDataCollator(pad_token_id=vocab['[PAD]'], max_length=max_length)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
                predictions = np.argmax(predictions, axis=-1)
            elif isinstance(predictions, torch.Tensor) and predictions.dim() == 3:
                predictions = predictions.argmax(dim=-1).cpu().numpy()
            predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
            decoded_preds = []
            decoded_labels = []
            for pred, label in zip(predictions, labels):
                pred_ids = [id for id in pred if id not in [vocab['[PAD]'], vocab['[CLS]'], vocab['[SEP]']]]
                label_ids = [id for id in label if id not in [vocab['[PAD]'], vocab['[CLS]'], vocab['[SEP]']]]
                decoded_preds.append([vocab.get(str(id), '[UNK]') for id in pred_ids])
                decoded_labels.append([vocab.get(str(id), '[UNK]') for id in label_ids])
            exact_match = sum(' '.join(p) == ' '.join(l) for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds) if decoded_preds else 0
            return {'exact_match': exact_match}

        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            vocab=vocab,
        )
        trainer.args.max_length = max_length

        if not os.path.exists(trainer.val_parts_dir) or not glob.glob(os.path.join(trainer.val_parts_dir, 'part_*')):
            trainer.split_val_dataset(val_dataset)

        logger.info("Starting training...")
        trainer.train()

        trainer.save_model('./final_model')
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train seq2seq model with optional quick validation")
    parser.add_argument('--quick-validate', action='store_true', help="Run quick validation with reduced dataset")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the training data directory")
    parser.add_argument('--start-folder', type=str, default="122334", help="First folder number (inclusive)")
    parser.add_argument('--end-folder', type=str, default="228474", help="Last folder number (inclusive)")
    args = parser.parse_args()

    logger.info(f"Environment ready, quick-validate mode: {args.quick_validate}")

    try:
        train(args.data_path, args.start_folder, args.end_folder, args.quick_validate)
    except Exception as e:
        logger.error(f"Main process failed: {e}")
        raise


if __name__ == "__main__":
    main()
