import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size=37_000, d_model=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._positional_encoding(100), persistent=False)
        self.max_seq_len = 100

    def _positional_encoding(self, seq_len, min_timescale=1, max_timescale=1e4):
        position = torch.arange(seq_len).float()
        num_timescales = self.d_model // 2
        log_timescale_increment = torch.log(
            torch.tensor(max_timescale) / torch.tensor(min_timescale)
        ) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, self.d_model % 2))
        return signal

    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            self.pe = self._positional_encoding(x.size(1))
            self.max_seq_len = x.size(1)
        return self.dropout(self.emb(x) + self.pe[: x.size(1)])


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__()
        assert total_key_depth % num_heads == 0, (
            "total_key_depth must be divisible by num_heads"
        )
        assert total_value_depth % num_heads == 0, (
            "total_value_depth must be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.key_depth_per_head = total_key_depth // num_heads
        self.dropout_rate = dropout_rate

        self.q_transform = nn.Linear(input_depth, total_key_depth)
        self.k_transform = nn.Linear(input_depth, total_key_depth)
        self.v_transform = nn.Linear(input_depth, total_value_depth)

        self.output_transform = nn.Linear(total_value_depth, output_depth)

        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def forward(self, query_antecedent, memory_antecedent=None, bias=None):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        assert query_antecedent.size(0) == memory_antecedent.size(0), (
            "query_antecedent and memory_antecedent must have the same batch size"
        )
        assert query_antecedent.size(2) == memory_antecedent.size(2), (
            "query_antecedent and memory_antecedent must have the same depth"
        )

        q = self._split_heads(self.q_transform(query_antecedent), self.num_heads)
        k = self._split_heads(self.k_transform(memory_antecedent), self.num_heads)
        v = self._split_heads(self.v_transform(memory_antecedent), self.num_heads)

        q = q / (self.key_depth_per_head**0.5)
        logits = torch.matmul(q, k.transpose(-2, -1))
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        output = self.output_transform(output)
        return output

    @staticmethod
    def _split_heads(x, num_heads):
        x = x.view(x.size(0), x.size(1), num_heads, x.size(2) // num_heads)
        return x.permute(0, 2, 1, 3)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                variance_scaling_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


class GlobalSelfAttention(MultiheadAttention):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__(
            input_depth=input_depth,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            output_depth=output_depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    def forward(self, query_antecedent, bias):
        return super().forward(query_antecedent, bias)


class CrossAttention(MultiheadAttention):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__(
            input_depth=input_depth,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            output_depth=output_depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    def forward(self, query_antecedent, memory_antecedent, bias):
        return super().forward(query_antecedent, memory_antecedent, bias=None)


class CausalSelfAttention(MultiheadAttention):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__(
            input_depth=input_depth,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            output_depth=output_depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    def forward(self, query_antecedent):
        return super().forward(
            query_antecedent,
            bias=self._generate_causal_mask(
                query_antecedent.size(1),
                query_antecedent.device,
                query_antecedent.dtype,
            ),
        )

    @staticmethod
    def _generate_causal_mask(size, device=None, dtype=None):
        return (
            torch.triu(torch.ones(size, size), diagonal=1).to(
                device=device, dtype=dtype
            )
            * -1e9
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        input_depth=512,
        output_depth=512,
        hidden_depth=2048,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_depth, hidden_depth)
        self.linear2 = nn.Linear(hidden_depth, output_depth)
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                variance_scaling_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        attention_dropout=0.1,
        relu_dropout=0.0,
        hidden_depth=2048,
        residual_dropout=0.1,
    ):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            input_depth,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            attention_dropout,
        )
        self.feed_forward = FeedForward(
            input_depth, output_depth, hidden_depth, relu_dropout
        )

        self.layer_norm_gsa = nn.LayerNorm(output_depth)
        self.layer_norm_ffn = nn.LayerNorm(output_depth)

        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x, bias):
        y = self.self_attention(x, bias)
        x = x + self.dropout(y)
        x = self.layer_norm_gsa(x)

        y = self.feed_forward(x)
        x = x + self.dropout(y)
        x = self.layer_norm_ffn(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        attention_dropout=0.1,
        relu_dropout=0.0,
        hidden_depth=2048,
        residual_dropout=0.1,
    ):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            input_depth,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            attention_dropout,
        )
        self.cross_attention = CrossAttention(
            input_depth,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            attention_dropout,
        )
        self.feed_forward = FeedForward(
            input_depth, output_depth, hidden_depth, relu_dropout
        )

        self.layer_norm_csa = nn.LayerNorm(output_depth)
        self.layer_norm_ca = nn.LayerNorm(output_depth)
        self.layer_norm_ffn = nn.LayerNorm(output_depth)

        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x, memory, bias):
        y = self.causal_self_attention(x)
        x = x + self.dropout(y)
        x = self.layer_norm_csa(x)

        y = self.cross_attention(x, memory, bias)
        x = x + self.dropout(y)
        x = self.layer_norm_ca(x)

        y = self.feed_forward(x)
        x = x + self.dropout(y)
        x = self.layer_norm_ffn(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        vocab_size=37_000,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        attention_dropout=0.1,
        relu_dropout=0.0,
        hidden_depth=2048,
        residual_dropout=0.1,
    ):
        super().__init__()
        self.embedding = PositionalEmbedding(vocab_size, input_depth, residual_dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    input_depth,
                    total_key_depth,
                    total_value_depth,
                    output_depth,
                    num_heads,
                    attention_dropout,
                    relu_dropout,
                    hidden_depth,
                    residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, bias):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, bias)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        vocab_size=37_000,
        input_depth=512,
        total_key_depth=512,
        total_value_depth=512,
        output_depth=512,
        num_heads=8,
        attention_dropout=0.1,
        relu_dropout=0.0,
        hidden_depth=2048,
        residual_dropout=0.1,
    ):
        super().__init__()
        self.embedding = PositionalEmbedding(vocab_size, input_depth, residual_dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    input_depth,
                    total_key_depth,
                    total_value_depth,
                    output_depth,
                    num_heads,
                    attention_dropout,
                    relu_dropout,
                    hidden_depth,
                    residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, memory, bias):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, memory, bias)
        return x


class Transformer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(
            num_layers=hparams.num_hidden_layers,
            vocab_size=hparams.vocab_size,
            input_depth=hparams.hidden_size,
            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
            output_depth=hparams.hidden_size,
            num_heads=hparams.num_heads,
            attention_dropout=hparams.attention_dropout,
            relu_dropout=hparams.relu_dropout,
            hidden_depth=hparams.filter_size,
            residual_dropout=hparams.residual_dropout,
        )

        self.decoder = Decoder(
            num_layers=hparams.num_hidden_layers,
            vocab_size=hparams.vocab_size,
            input_depth=hparams.hidden_size,
            total_key_depth=hparams.attention_key_channels or hparams.hidden_size,
            total_value_depth=hparams.attention_value_channels or hparams.hidden_size,
            output_depth=hparams.hidden_size,
            num_heads=hparams.num_heads,
            attention_dropout=hparams.attention_dropout,
            relu_dropout=hparams.relu_dropout,
            hidden_depth=hparams.filter_size,
            residual_dropout=hparams.residual_dropout,
        )

        self.final_layer = nn.Linear(hparams.hidden_size, hparams.vocab_size)

    def forward(self, src, src_mask, tgt):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask)
        logits = self.final_layer(decoder_output)
        return logits


def variance_scaling_uniform_(tensor, gain=1.0, mode="fan_avg"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid mode for variance scaling initializer.")

    scale = gain / max(1.0, denom)
    limit = (3.0 * scale) ** 0.5
    return nn.init.uniform_(tensor, -limit, limit)
