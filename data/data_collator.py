import torch
from typing import Dict, List

from dataclasses import dataclass
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorWithPaddingForHaRT:
    """
        Data collator that simply collates batches of lists of dict-like objects 
        and adds padding where none.
        Also, sets other model inputs if passed in args.

        """

    def __init__(self, model_args, config, tokenizer: PreTrainedTokenizerBase, deepspeed=False, is_ft=False, is_user_level_ft=False):
        self.is_ft = is_ft
        self.is_user_level_ft = is_user_level_ft
        self.tokenizer = tokenizer
        self.output_block_last_hidden_states = None if is_ft else model_args.output_block_last_hidden_states 
        if model_args.add_history or config.add_history:
            self.history = torch.load(model_args.initial_history) if model_args.initial_history else (torch.zeros(config.n_embd))
            self.history = self.history.to(torch.float16) if deepspeed else self.history.float() 
            if not is_ft:
                self.layer_ins = model_args.layer_ins if model_args.layer_ins else config.layer_ins
                self.extract_layer = model_args.extract_layer if model_args.extract_layer else config.extract_layer
        else:
            self.history = None 
            self.layer_ins = None
            self.extract_layer = None       

    def __call__(self, examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
        # In this function we'll make the assumption that all `examples` in the batch of lists
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # in the whole batch of lists
        if not isinstance(examples[0], list) or \
        (not self.is_user_level_ft and not isinstance(examples[0][0], (dict, BatchEncoding))) or \
        (self.is_user_level_ft and not isinstance(examples[0][2], (dict, BatchEncoding))):
            raise ValueError("You landed on an incorrect collator! This one's AR_HuLM specific.")

        first = examples[0][2] if self.is_user_level_ft else examples[0][0]
        batch = {}

        if self.is_user_level_ft:
            batch['user_ids'] = torch.tensor([
                                            example[0]
                                            for example in examples
                                            ])
            batch['labels'] = torch.tensor([
                                            example[1]
                                            for example in examples
                                            ])
            # we do this to map it to the examples format as received when not user_level_ft,
            # in order to reuse the rest of the following code for data collation
            blocks = [example[2:] for example in examples] 
            examples = blocks 
        
        

        # Handling all possible keys as figured from the first element        
        for k, v in first.items():
            if k not in ("input_ids", "attention_mask", "labels"):
                raise ValueError("You've landed at an incorrect collator! This one's specific to AR_HuLM.")

            if v is not None and not isinstance(v, str):
                pad = self.tokenizer.eos_token_id if k=='input_ids' else 0 if k=='attention_mask' else -100 
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples]) 
                else:
                    # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                    batch[k] = torch.tensor([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples
                                            ]) 
        
        block_size = len(first['input_ids'])
        batch['history'] = None if self.history is None else self.history.repeat(len(examples), block_size, 1)
        if not self.is_ft:
            batch['layer_ins'] = self.layer_ins
            batch['extract_layer'] = self.extract_layer
            batch['output_block_last_hidden_states'] = self.output_block_last_hidden_states

        return batch 

@dataclass
class HaRTDefaultDataCollator:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential mappings of user_ids and text_ids to the model inputs.:

        - ``text_ids``: handles a list value per object
        - ``user_ids``: handles a single value per object

    Preprocess padded blocks with pad tokens. Property names of the input object will be used as corresponding inputs
    to the model. Initialized initial user state (history) with the default used for HaRT model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        initial_history_path = Path(__file__).parent / "initial_history/initialized_history_tensor.pt"

        self.history = torch.load(initial_history_path).float()

    
    def __call__(self, examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
        
            # In this function we'll make the assumption that all `examples` in the batch of lists
            # have the same attributes.
            # So we will look at the first element as a proxy for what attributes exist
            # in the whole batch of lists
        if not isinstance(examples[0], list) or \
            not isinstance(examples[0][2], (dict, BatchEncoding)):
                raise ValueError("You landed on an incorrect collator! This one's AR_HuLM specific.")
                return
        

        first = examples[0][2]
        batch = {}
        
        batch['user_ids'] = [
                            example[0]
                            for example in examples
                            ]
        batch['text_ids'] = [
                            example[1]
                            for example in examples
                            ]
        # we do this to map it to the examples format as received when not user_level_ft,
        # in order to reuse the rest of the following code for data collation
        blocks = [example[2:] for example in examples] 
        examples = blocks 
        
        
        # Handling all possible keys as figured from the first element        
        for k, v in first.items():
            if k not in ("input_ids", "attention_mask", "labels"):
                raise ValueError("You've landed at an incorrect collator! This one's specific to AR_HuLM.")
            if v is not None and not isinstance(v, str):
                pad = self.tokenizer.eos_token_id if k=='input_ids' else 0 if k=='attention_mask' else -100 
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples]) 
                else:
                    # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                    batch[k] = torch.tensor([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples
                                            ])
                    
        block_size = len(first['input_ids'])
        batch['history'] = None if self.history is None else self.history.repeat(len(examples), block_size, 1)
        
        return batch

def user_default_data_collator(examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``labels``: handles a single value (int or float) per object
        - ``user_id``: handles a single value per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. 
    """

        # In this function we'll make the assumption that all `examples` in the batch of lists
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # in the whole batch of lists
    if not isinstance(examples[0], list) or \
        not isinstance(examples[0][2], (dict, BatchEncoding)):
            raise ValueError("You landed on an incorrect collator! This one's AR_HuLM specific.")
            return
    

    first = examples[0][2]
    batch = {}
    
    batch['user_ids'] = torch.tensor([
                                    example[0]
                                    for example in examples
                                    ])
    batch['labels'] = torch.tensor([
                                    example[1]
                                    for example in examples
                                    ])
    # we do this to map it to the examples format as received when not user_level_ft,
    # in order to reuse the rest of the following code for data collation
    blocks = [example[2:] for example in examples] 
    examples = blocks 
    
    
    # Handling all possible keys as figured from the first element        
    for k, v in first.items():
        if k not in ("input_ids", "attention_mask", "labels"):
            raise ValueError("You've landed at an incorrect collator! This one's specific to AR_HuLM.")
            return
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([be[k] for example in examples for be in example]) 
            else:
                # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                batch[k] = torch.tensor([be[k] for example in examples for be in example])
    
    return batch 