import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    """
    LSTM-based decoder that generates captions from encoded image features.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        #self.lstm = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout, batch_first=True)
        #self.norm = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, vocab_size)


    def forward(self, features, captions):
        embeddings = self.embed(captions)

        embeddings[:, 0, :] = features #replace first elem of each seq,<start>, by corresponding image embedding

        outputs, _ = self.lstm(embeddings)
        outputs = self.linear(outputs)
        return outputs

    def forward_inference_step(self, input_token_batch, state_info_batch, features_emb=None):
        """
        Used during inference — one step at a time.
        Input - (N, 1), where N - batch size, and (1) is seq of len=1 -> token to pass next.
        """
        emb = self.embed(input_token_batch)
        if features_emb is not None:
            emb = features_emb.unsqueeze(1)

        output, state_info = self.lstm(emb, state_info_batch)
        output = self.linear(output)
        return output, state_info
