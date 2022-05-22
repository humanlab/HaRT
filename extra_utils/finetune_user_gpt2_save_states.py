from more_itertools import split_at

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MSELoss

from transformers.utils import logging

from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


logger = logging.get_logger(__name__)

class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config, tokenizer, args, save_user_states=False):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.save_user_states = save_user_states
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)

        # self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # self.transform = nn.Linear(config.n_embd, config.n_embd)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def process_into_messages(self, data):
        def pad_message(data, pad_value):
            for i,x in enumerate(data):
                if len(x) > 200:
                    x = x[0:200]
                    data[i] = x
                else:
                    x.extend([pad_value]*(200-len(x)))
            return data
            
        def pad_and_collate_data(data):
            i_values = data
            a_values = [[1]*len(x) for x in i_values]
            
            i_values = pad_message(i_values, self.tokenizer.eos_token_id)
            a_values = pad_message(a_values, 0)

            return i_values, a_values

        def split_into_messages(data):
            i_values = data.reshape(-1).tolist()
            i_values = list(split_at(i_values, lambda x:x==self.tokenizer.eos_token_id))[0]
            i_values = i_values[:-1] if i_values[-1]==self.tokenizer.sep_token_id else i_values
            i_values = list(split_at(i_values, lambda x:x==self.tokenizer.sep_token_id))
            return i_values

        input_ids = split_into_messages(data)
        input_ids, attention_mask = pad_and_collate_data(input_ids)
        return input_ids, attention_mask, len(input_ids)

    def _prepare_inputs(self, input):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting tensors to cuda tensors
        """
        
        if isinstance(input, torch.Tensor):
            input = input.to(self.args.device)
        return input

    def get_user_embedding(self, hidden_states, attention_mask, user_num_msgs):
        user_hs_splits = torch.split(hidden_states, user_num_msgs)
        user_attn_splits = torch.split(attention_mask, user_num_msgs)
        assert len(user_hs_splits) == len(user_attn_splits)
        user_states = None
        for states, masks in zip(user_hs_splits, user_attn_splits):
            masks = masks.unsqueeze(dim=2)
            masked_states = states*masks
            summed_msg_hs = torch.sum(masked_states, dim=1)
            summed_msg_masks = torch.sum(masks, dim=1)
            message_states = summed_msg_hs/summed_msg_masks
            summed_user_states = torch.sum(summed_msg_hs, dim=0)
            num_msgs = message_states.shape[0]
            averaged_user_states = summed_user_states/num_msgs
            if user_states is None:
                user_states = averaged_user_states.unsqueeze(dim=0)
            else:
                user_states = torch.cat([user_states, averaged_user_states.unsqueeze(dim=0)])
        return user_states

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
        user_num_msgs = []
        if user_ids is not None:
            user_messages = torch.Tensor()
            user_msg_attn = torch.Tensor()
            for u_input_ids in input_ids:
                ids, attn, num_msgs = self.process_into_messages(u_input_ids)
                user_num_msgs.append(num_msgs)
                ids = torch.Tensor(ids)
                attn = torch.Tensor(attn)
                user_messages = torch.cat([user_messages, ids])
                user_msg_attn = torch.cat([user_msg_attn, attn])

            input_ids = self._prepare_inputs(user_messages).long()
            attention_mask = self._prepare_inputs(user_msg_attn).long()

        transformer_outputs = self.transformer(
            input_ids,
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
        hidden_states = transformer_outputs[0]
        # hidden_states = self.transform(self.ln_f(hidden_states))

        if user_ids is not None:
            user_states = self.get_user_embedding(hidden_states, attention_mask, user_num_msgs)

        if self.save_user_states:
            users = pd.DataFrame(user_ids.cpu().detach().numpy(), columns=['user_id'])
            users = users.loc[users.index.repeat(self.config.embed)]
            users.reset_index(drop=True, inplace=True)

            user_states = pd.DataFrame(user_states.cpu().detach().numpy())
            user_states = user_states.stack().reset_index()
            user_states['level_0'] = users['user_id']
            user_states.rename(columns={'level_0':'user_id','level_1': 'column_number', 0:'value'}, inplace=True)

            user_states.to_csv(self.args.output_dir + '/test_states_' + str(user_ids[0].item()) + '.csv', index=False)

            logits = user_states
            loss = loss = torch.Tensor([0.1]).cuda()        
        else:
            # logits = self.score(self.transform(self.ln_f(hidden_states)))
            logits = self.score(user_states)

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.to(self.dtype).view(-1))
                else:
                    raise ValueError("You're in the wrong finetuner!")

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
