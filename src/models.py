

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel,
    RobertaModel
)

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.cur_train_epoch = 0
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.deterministic = config.deterministic
        self.ib_dim = config.ib_dim
        self.ib = config.ib
        self.activation = config.activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if self.ib or self.deterministic:
            self.kl_annealing = config.kl_annealing
            self.hidden_dim = config.hidden_dim
            intermediate_dim = (self.hidden_dim+config.hidden_size)//2
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = config.beta
            self.sample_size = config.sample_size
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))

            self.classifier = nn.Linear(self.ib_dim, config.num_labels)

        else:
            self.classifier = RobertaClassificationHead(config)

        self.post_init()

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid":
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits


    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        sampling_type="iid",
        during_train=True,
    ):
        self.cur_train_epoch += 1
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        loss = None

        if self.deterministic:
            pooled_output = self.mlp(pooled_output)
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            sampled_logits, logits = self.get_logits(mu, mu, sampling_type='argmax') # always deterministic
            if labels is not None:
                loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type='argmax')

        elif self.ib:
            pooled_output = self.mlp(pooled_output)
            batch_size = pooled_output.shape[0]
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
            kl_loss = self.kl_div(mu, std, mu_p, std_p)
            z = self.reparameterize(mu, std)

            if self.kl_annealing == "linear":
                if during_train:
                    beta = min(1.0, self.cur_train_epoch*self.beta)
                else:
                    beta = min(1.0, self.beta)
                 
            sampled_logits, logits = self.get_logits(z, mu, sampling_type)
            if labels is not None:
                ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
                loss = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * kl_loss
        else:
            logits = self.classifier(sequence_output)
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )