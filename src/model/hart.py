import torch
import torch.nn as nn

from src.model.modeling_hart import HaRTBasePreTrainedModel, HaRTBaseLMHeadModel
from src.modeling_outputs import HaRTOutput
from transformers.activations import ACT2FN

""" HaRT model pre-trained for the HuLM task """,

class HistoryMLP(nn.Module):
    def __init__(self, n_state, config):  # in history MLP: n_state=200
        super().__init__()
        nx = config.n_embd
        self.config = config
        self.l_hist = nn.Linear(nx, nx)
        self.l_hs = nn.Linear(nx, nx)
        self.act = ACT2FN["tanh"]

        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, history, hidden_state, sequence_mask):
        h1 = self.l_hist(history)
        h2 = self.l_hs(hidden_state)

        # Fixing the bug where sequence length is -1 when all tokens are padded (i.e. attn_mask is all zeros)
        h2 = h2 * sequence_mask
        # expand along block_len dimension (1) to allow addition with history
        h2 = h2.unsqueeze(1) # [batch_size, 1, embed_dim]

        return self.act(h1 + h2) # [batch_size, block_size, embed_dim]

class HaRTPreTrainedModel(HaRTBasePreTrainedModel):
    def __init__(self, config, hartbaseLMmodel=None):
        super().__init__(config)
        self.config = config
        inner_dim = config.n_inner if config.n_inner is not None else 200
        if hartbaseLMmodel:
            self.transformer = hartbaseLMmodel
        else:
            self.transformer = HaRTBaseLMHeadModel(config)
        if config.add_history:
            self.history_mlp = HistoryMLP(inner_dim, config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
      
    def get_last_pred_token_hidden_state(self, hs, attn_mask):
        batch_size = attn_mask.shape[0]
        
        # finds the last token that is not a padding token in each row.
        sequence_lengths = torch.ne(attn_mask, 0).sum(-1) - 1  # [batch_size]
        
        # selects the indices in sequence_lengths for the respective row in batch, i.e, 
        # finds the embedding of the last non-padded token (indices from sequence_lengths) in each row
        last_pred_token_hs = hs[range(batch_size), sequence_lengths] # [batch_size, embed_dim]
        
        # Fixing the bug where sequence length is -1 when all tokens are padded (i.e. attn_mask is all zeros)
        sequence_mask = (sequence_lengths != -1).int()
        sequence_mask = sequence_mask.unsqueeze(1)
    
        return last_pred_token_hs, sequence_mask

    def forward(
        self,
        input_ids=None,
        history=None,
        layer_ins=None,
        extract_layer=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_block_last_hidden_states=None,
        output_block_extract_layer_hs=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        layer_ins = layer_ins if layer_ins else self.config.layer_ins
        extract_layer = extract_layer if extract_layer else self.config.extract_layer

        usr_seq_len, blocks_len, block_size = input_ids.shape
        batch_loss = torch.tensor(0.0).to(self.device)
        batch_len = 0
        all_blocks_last_hs = () if output_block_last_hidden_states else None
        all_blocks_history = ()
        all_blocks_attn_mask = ()
        all_blocks_extract_layer_hs = ()

        for i in range(blocks_len):
            block_input_ids = input_ids[:,i,:]
            block_attention_mask = attention_mask[:,i,:]
            block_labels = labels[:,i,:] if labels is not None else None

            arhulm_output = self.transformer(
                    input_ids=block_input_ids,
                    history=history,
                    layer_ins=layer_ins,
                    extract_layer=extract_layer,
                    past_key_values=past_key_values,
                    attention_mask=block_attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=block_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
            last_block_last_hs = arhulm_output.last_hidden_state
            
            if output_block_last_hidden_states:
                all_blocks_last_hs = all_blocks_last_hs + (arhulm_output.last_hidden_state,)
            
            extract_layer_hs = arhulm_output["extract_layer_hidden_states"][0] if isinstance(arhulm_output, dict) else arhulm_output[-1][0] 
            
            if output_block_extract_layer_hs:
                all_blocks_extract_layer_hs = all_blocks_extract_layer_hs + (extract_layer_hs, )

            if history is not None:
                hs, sequence_mask = self.get_last_pred_token_hidden_state(extract_layer_hs, block_attention_mask)
                history = self.history_mlp(history, hs, sequence_mask)
                all_blocks_history = all_blocks_history + (history[:, 0, :],)
                all_blocks_attn_mask = all_blocks_attn_mask + (sequence_mask, )
                

            if labels is not None:
                batch_loss += arhulm_output["loss"] if isinstance(arhulm_output, dict) else arhulm_output[0] 
                batch_len += len(block_labels[block_labels!= -100])       

        loss = batch_loss/batch_len if labels is not None else None

        last_updated_history = history
        history_output = (all_blocks_history, all_blocks_attn_mask)
     
        if not return_dict:
            output = (last_block_last_hs, last_block_last_hs,) + arhulm_output[3:]
            return ((loss,) + output) if loss is not None else output

        return HaRTOutput(
            loss=loss,
            last_hidden_state=last_block_last_hs,
            all_blocks_last_hidden_states = all_blocks_last_hs,
            all_blocks_extract_layer_hs = all_blocks_extract_layer_hs,
            history=history_output,
            past_key_values=arhulm_output.past_key_values,
            hidden_states=arhulm_output.hidden_states,
            attentions=arhulm_output.attentions,
            cross_attentions=arhulm_output.cross_attentions,
        )
