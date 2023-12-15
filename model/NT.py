import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

class NTModel(nn.Module):
    def __init__(self, config, num_features):
        super(NTModel, self).__init__()
        self.num_features = num_features
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.dense = nn.Linear(num_features, config.hidden_size)
        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(features[:, :, 0])

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        features_transformed = self.dense(features)
        features_transformed = self.LayerNorm(features_transformed)
        features_transformed = self.dropout(features_transformed)

        encoded_layers = self.encoder(features_transformed,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertForSequenceClassification(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_features):
        super(BertForSequenceClassification, self).__init__(config)
        self.nt_model = NTModel(config, num_features=num_features)
        self.classifier = nn.Linear(config.hidden_size, 1)  # Binary classification

    def forward(self, features, attention_mask=None, labels=None):
        sequence_output, _ = self.nt_model(features, attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(sequence_output).squeeze(-1)  # Squeeze to remove last dimension

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, logits
        else:
            return logits

# Example usage:
# config = ... (load or define your BertConfig here)
# num_features = 4  # Number of features (HR, Temp, Spo2, Resp)
# model = BertForSequenceClassification(config, num_features)
# Then use this model for training/evaluation with your NTLoader