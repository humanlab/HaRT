import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from .modeling_hart import HaRTBasePreTrainedModel
from .hart import HaRTPreTrainedModel

class HaRTForSequenceClassification(HaRTBasePreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config, model_name_or_path=None, pt_model=None):
        super().__init__(config)
        self.freeze_model = config.freeze_model
        self.num_labels = config.num_labels
        self.finetuning_task = config.finetuning_task
        self.use_history_output = config.use_history_output
        self.use_hart_no_hist = config.use_hart_no_hist
        if model_name_or_path:
            self.transformer = HaRTPreTrainedModel.from_pretrained(model_name_or_path)
        elif pt_model:
            self.transformer = pt_model
        else:
            self.transformer = HaRTPreTrainedModel(config)
            self.init_weights()
        
        if not self.freeze_model and not self.finetuning_task=='ope' and not self.finetuning_task=='user':
            self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if self.finetuning_task=='age':
            self.transform = nn.Linear(config.n_embd, config.n_embd)

        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_pooled_logits(self, logits, input_ids, inputs_embeds):

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                # since we want the index of the last predicted token of the last block only.
                sequence_lengths = sequence_lengths[:, -1]
            else:
                sequence_lengths = -1
                self.logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # get the logits corresponding to the indices of the last pred tokens (of the last blocks) of each user
        pooled_logits = logits[range(batch_size), sequence_lengths]

        return pooled_logits

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
            output_block_extract_layer_hs=True,
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

        all_blocks_last_hidden_states = transformer_outputs.all_blocks_extract_layer_hs if self.freeze_model else transformer_outputs.all_blocks_last_hidden_states 
        
        if self.finetuning_task=='user' or self.finetuning_task=='ope' or self.finetuning_task=='age':
            if self.use_history_output:
                states = transformer_outputs.history[0]
                masks = transformer_outputs.history[1]
                multiplied = tuple(l * r for l, r in zip(states, masks))
                all_blocks_user_states = torch.stack(multiplied, dim=1)
                all_blocks_masks = torch.stack(masks, dim=1)
                sum = torch.sum(all_blocks_user_states, dim=1)
                divisor = torch.sum(all_blocks_masks, dim=1)
                hidden_states = sum/divisor
            else:
                raise ValueError("Since you don't want to use the user-states/history output for a user-level task, please customize the code as per your requirements.")
        else:
            hidden_states = torch.stack(all_blocks_last_hidden_states, dim=1)
       
        if self.use_hart_no_hist:
            logits = self.score(all_blocks_last_hidden_states[0]) if self.freeze_model else self.score(self.ln_f(all_blocks_last_hidden_states[0]))
            batch_size, _, sequence_length = input_ids.shape
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            pooled_logits = logits[range(batch_size), sequence_lengths.squeeze()]
        else:
            if self.finetuning_task=='ope' or self.finetuning_task=='user' or self.freeze_model:
                logits = self.score(hidden_states) 
            elif self.finetuning_task=='age':
                logits = self.score(self.transform(self.ln_f(hidden_states)))
            else:
                logits = self.score(self.ln_f(hidden_states))
            pooled_logits = logits if (user_ids is None or self.use_history_output) else \
                        self.get_pooled_logits(logits, input_ids, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
            else:
                labels = labels.long()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
