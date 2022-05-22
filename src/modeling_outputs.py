from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.file_utils import ModelOutput

@dataclass
class HaRTOutput(ModelOutput):
    """
    Superset of ArHulmCausalLMOutput with history in the output additionally. 

    The description of ArHuLMOutput is as follows which also has copied description of CausalLMOutputWithCrossAttentions from Hugging Face's transformers.modeling_outputs.py:

    Base class for Auto regressive Human language model outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        # logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
        #     Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.            
        all_blocks_last_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_block_last_hidden_states=True`` is passed or when ``config.output_block_last_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of each block's last hidden state)
            of shape :obj:`(batch_size, blocks_length, sequence_length, hidden_size)`.

            Last hidden-states of the model's blocks at the output of last layer for each block.
        all_blocks_extract_layer_hs (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_block_last_hidden_states=True`` is passed or when ``config.output_block_last_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of each block's last hidden state)
            of shape :obj:`(batch_size, blocks_length, sequence_length, hidden_size)`.

            Extract Layer's hidden-states of the model's blocks at the output of last layer for each block.
        history (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. where each sequence length for a respective batch instance will have the same hidden_embeds.

            Residual history of the users in the batch by the end of the model processing after applying recurrence throughout.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`torch.FloatTensor` tuples of length :obj:`config.n_layers`, with each tuple containing the
            cached key, value states of the self-attention and the cross-attention layers if model is used in
            encoder-decoder setting. Only relevant if ``config.is_decoder = True``.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        # extract_layer_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``extract_layer is not None`` (i.e, a value is passed) or when ``config.extract_layer has a value other than None``):
        #     Tuple of :obj:`torch.FloatTensor` (one for the output of (each) extract layer) -- currently takes in 1 value of extract_layer
        #     of shape :obj:`(batch_size, sequence_length, hidden_size)`.

        #     Hidden-states of the model at the output of extract layer.
    """

    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    history: torch.FloatTensor = None
    all_blocks_last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_blocks_extract_layer_hs: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class HaRTBaseModelOutput(ModelOutput):
    """
    Overridden BaseModelOutputWithPastAndCrossAttentions to add extract_layer hidden states to the output for AR HuLM model.

    The description of BaseModelOutputWithPastAndCrossAttentions is as follows copied from Hugging Face's transformers.modeling_outputs.py:
    [Adding an arg 'extract_layer_hidden_states' in the description]

    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        extract_layer_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``extract_layer is not None`` (i.e, a value is passed) or when ``config.extract_layer has a value other than None``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of (each) extract layer) -- currently takes in 1 value of extract_layer
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of extract layer.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    extract_layer_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HaRTBaseCausalLMOutput(ModelOutput):
    """
    Overridden CausalLMOutputWithCrossAttentions to add extract_layer hidden states to the output for AR HuLM model.

    The description of CausalLMOutputWithCrossAttentions is as follows copied from Hugging Face's transformers.modeling_outputs.py:
    [Adding an arg 'extract_layer_hidden_states' in the description]

    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`torch.FloatTensor` tuples of length :obj:`config.n_layers`, with each tuple containing the
            cached key, value states of the self-attention and the cross-attention layers if model is used in
            encoder-decoder setting. Only relevant if ``config.is_decoder = True``.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        extract_layer_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``extract_layer is not None`` (i.e, a value is passed) or when ``config.extract_layer has a value other than None``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of (each) extract layer) -- currently takes in 1 value of extract_layer
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of extract layer.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    extract_layer_hidden_states: Optional[Tuple[torch.FloatTensor]] = None