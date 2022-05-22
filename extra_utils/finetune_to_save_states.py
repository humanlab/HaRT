import pandas as pd
import torch
import torch.nn as nn

from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from ..src.model.hart import HaRTPreTrainedModel

class ArHulmForSequenceClassification(HaRTPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config, output_dir, agg_type, arhulm=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_history_output = config.use_history_output
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.output_dir = output_dir
        self.agg_type = agg_type
        if arhulm:
            self.transformer = arhulm
        else:
            self.transformer = HaRTPreTrainedModel(config)
            self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        user_ids=None,
        history=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            history=history,
            output_block_last_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        users = pd.DataFrame(user_ids.cpu().detach().numpy(), columns=['user_id'])
        users = users.loc[users.index.repeat(768)]
        users.reset_index(drop=True, inplace=True)

        if self.agg_type=='last':
            user_states = transformer_outputs.history[0][-1]
        elif self.agg_type=='sum':
            all_blocks_user_states = torch.stack(transformer_outputs.history[0], dim=1)
            user_states = torch.sum(all_blocks_user_states, dim=1)
        elif self.agg_type=='avg':
            all_blocks_user_states = torch.stack(transformer_outputs.history[0], dim=1)
            user_states = torch.sum(all_blocks_user_states, dim=1)/all_blocks_user_states.shape[1]
        elif self.agg_type=='masked_last':
            states = transformer_outputs.history[0]
            masks = transformer_outputs.history[1]
            multiplied = tuple(l * r for l, r in zip(states, masks))
            all_blocks_user_states = torch.stack(multiplied, dim=1).cpu().detach()
            all_blocks_masks = torch.stack(masks, dim=1)
            divisor = torch.sum(all_blocks_masks, dim=1).cpu().detach()
            user_states = all_blocks_user_states[range(all_blocks_user_states.shape[0]), divisor.squeeze()-1]
        elif self.agg_type=='masked_sum':
            states = transformer_outputs.history[0]
            masks = transformer_outputs.history[1]
            multiplied = tuple(l * r for l, r in zip(states, masks))
            all_blocks_user_states = torch.stack(multiplied, dim=1)
            user_states = torch.sum(all_blocks_user_states, dim=1)
        elif self.agg_type=='masked_avg':
            states = transformer_outputs.history[0]
            masks = transformer_outputs.history[1]
            multiplied = tuple(l * r for l, r in zip(states, masks))
            all_blocks_user_states = torch.stack(multiplied, dim=1)
            all_blocks_masks = torch.stack(masks, dim=1)
            sum = torch.sum(all_blocks_user_states, dim=1)
            divisor = torch.sum(all_blocks_masks, dim=1)
            user_states = sum/divisor
        logits = user_states
        loss = torch.Tensor([0.1]).cuda()
        user_states = pd.DataFrame(user_states.cpu().detach().numpy())
        user_states = user_states.stack().reset_index()
        user_states['level_0'] = users['user_id']
        user_states.rename(columns={'level_0':'user_id','level_1': 'column_number', 0:'value'}, inplace=True)

        user_states.to_csv(self.output_dir + '/test_states_' + str(user_ids[0].item()) + '.csv', index=False)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
