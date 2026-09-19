import torch
from torch import nn
from transformers import LongformerConfig


class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_frequencies=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies

        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(num_frequencies))
        self.b = nn.Parameter(torch.randn(num_frequencies))
        self.projection = nn.Linear(num_frequencies + 1, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        linear_out = self.w0 * x + self.b0
        periodic_out = torch.sin(self.w * x + self.b)
        return self.projection(torch.cat([linear_out, periodic_out], dim=-1))


class ValueEmbedding(nn.Module):
    """Simple value-embedding baseline. Input x must already be standardized."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.value_embedding(x.to(torch.float32).unsqueeze(-1))


class ContinuousValueEmbedding(nn.Module):
    """
    Normalized multi-view gated continuous-value embedding.

    values must be standardized BEFORE masking:
        z_ij = (x_ij - mean_j) / (std_j + eps)

    mean_j/std_j must be estimated from PRETRAIN TRAIN only and then frozen.

    Three retained views:
      1. standardized linear view
      2. nonlinear MLP view
      3. signed-log compressed view

    This preserves the original multi-view gated value-representation idea while
    removing the raw-scale dependence criticized in value prediction.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.standardized_projection = nn.Linear(1, embedding_size)

        self.value_transform = nn.Sequential(
            nn.Linear(1, embedding_size * 2),
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
        )

        self.log_scale = nn.Parameter(torch.ones(1))
        self.log_projection = nn.Linear(1, embedding_size)

        self.gate = nn.Linear(embedding_size * 3, 3)
        self.final_projection = nn.Linear(embedding_size, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

        self._init_weights()

    def _init_weights(self):
        for layer in self.value_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in (
            self.standardized_projection,
            self.log_projection,
            self.gate,
            self.final_projection,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values):
        values = values.to(torch.float32).unsqueeze(-1)

        standardized_embeds = self.standardized_projection(values)
        transformed_embeds = self.value_transform(values)

        signed_log_values = (
            torch.sign(values)
            * torch.log1p(torch.abs(values))
            * self.log_scale
        )
        log_embeds = self.log_projection(signed_log_values)

        combined_embeds = torch.cat(
            [standardized_embeds, transformed_embeds, log_embeds],
            dim=-1,
        )
        gate_weights = torch.sigmoid(self.gate(combined_embeds))

        gated_output = (
            gate_weights[..., 0:1] * standardized_embeds
            + gate_weights[..., 1:2] * transformed_embeds
            + gate_weights[..., 2:3] * log_embeds
        )

        return self.layer_norm(self.final_projection(gated_output))


class UnitEmbedding(nn.Module):
    def __init__(self, unit_size, embedding_size):
        super().__init__()
        self.unit_embedding = nn.Embedding(unit_size, embedding_size)

    def forward(self, x):
        return self.unit_embedding(x)


class AgeEmbedding(nn.Module):
    def __init__(self, max_age, embedding_size):
        super().__init__()
        self.age_embedding = nn.Embedding(max_age, embedding_size)

    def forward(self, x):
        return self.age_embedding(x)


class GenderEmbedding(nn.Module):
    def __init__(self, gender_size, embedding_size):
        super().__init__()
        self.gender_embedding = nn.Embedding(gender_size, embedding_size)

    def forward(self, x):
        return self.gender_embedding(x)


class TaskEmbedding(nn.Module):
    def __init__(self, task_size, embedding_size):
        super().__init__()
        self.task_embedding = nn.Embedding(task_size, embedding_size)

    def forward(self, x):
        return self.task_embedding(x)


class EHRtokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)


def _get_value_embedding_type(args):
    if args is None:
        return "continuous"
    return getattr(args, "value_embedding_type", "continuous")


class EHREmbedding(nn.Module):
    """
    Pretraining embedding compatible with the new Dataset.

    Event representation:
        concept + time + normalized value + unit

    Prepended special representations:
        task + age + gender

    Removed:
        position embedding
        order-category-name embedding
        order-category-description embedding

    token_type_ids remains in the signature because it is still useful for
    event-type-specific pretraining masking, but it is not embedded here.
    """

    def __init__(
        self,
        config: LongformerConfig,
        itemid_size,
        unit_size,
        max_age,
        gender_size,
        task_size,
        token_type_size=None,
        padding_idx=0,
        use_itemid=True,
        inputs_embeds=None,
        args=None,
        **legacy_kwargs,
    ):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.itemid_size = itemid_size
        self.unit_size = unit_size
        self.max_age = max_age
        self.hidden_size = config.hidden_size
        self.gender_size = gender_size
        self.task_size = task_size
        self.token_type_size = token_type_size
        self.padding_idx = padding_idx
        self.use_itemid = use_itemid
        self.inputs_embeds = inputs_embeds
        self.args = args

        self.concept_embedding = EHRtokenEmbedding(
            self.itemid_size, self.hidden_size
        )
        self.time_embedding = TimeEmbedding(1, self.hidden_size)

        if _get_value_embedding_type(args) == "simple":
            self.value_embedding = ValueEmbedding(self.hidden_size)
        else:
            self.value_embedding = ContinuousValueEmbedding(self.hidden_size)

        self.unit_embedding = UnitEmbedding(
            self.unit_size, self.hidden_size
        )
        self.age_embedding = AgeEmbedding(
            self.max_age, self.hidden_size
        )
        self.gender_embedding = GenderEmbedding(
            self.gender_size, self.hidden_size
        )
        self.task_embedding = TaskEmbedding(
            self.task_size, self.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        value_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        time_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        age_ids: torch.Tensor,
        gender_ids: torch.Tensor,
        task_ids: torch.Tensor,
        inputs_embeds=None,
        **legacy_kwargs,
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        concept_embed = self.concept_embedding(input_ids)

        time_embed = (
            self.time_embedding(time_ids)
            if time_ids is not None
            else torch.zeros_like(concept_embed)
        )
        value_embed = (
            self.value_embedding(value_ids)
            if value_ids is not None
            else torch.zeros_like(concept_embed)
        )
        unit_embed = (
            self.unit_embedding(unit_ids)
            if unit_ids is not None
            else torch.zeros_like(concept_embed)
        )

        age_embed = self.age_embedding(age_ids)
        gender_embed = self.gender_embedding(gender_ids)
        task_embed = self.task_embedding(task_ids)

        event_embeddings = (
            concept_embed
            + time_embed
            + value_embed
            + unit_embed
        )

        return torch.cat(
            (task_embed, age_embed, gender_embed, event_embeddings),
            dim=1,
        )


class EHREmbedding_finetune(nn.Module):
    """
    Finetuning embedding aligned with EHREmbedding.

    Supported feature ablations after permanently removing position/order:
        time, value, unit
    """

    def __init__(
        self,
        config: LongformerConfig,
        itemid_size,
        unit_size,
        max_age,
        gender_size,
        task_size,
        token_type_size=None,
        ablation=None,
        padding_idx=0,
        use_itemid=True,
        inputs_embeds=None,
        args=None,
        **legacy_kwargs,
    ):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.itemid_size = itemid_size
        self.unit_size = unit_size
        self.max_age = max_age
        self.hidden_size = config.hidden_size
        self.gender_size = gender_size
        self.task_size = task_size
        self.token_type_size = token_type_size
        self.padding_idx = padding_idx
        self.use_itemid = use_itemid
        self.inputs_embeds = inputs_embeds
        self.args = args

        if ablation is None:
            self.ablation = []
        elif isinstance(ablation, str):
            self.ablation = [x for x in ablation.split("+") if x]
        elif isinstance(ablation, list):
            self.ablation = list(ablation)
        else:
            raise ValueError("ablation must be None, a string, or a list.")

        allowed_ablation = {"time", "value", "unit"}
        invalid = set(self.ablation) - allowed_ablation
        if invalid:
            raise ValueError(
                f"Unsupported ablation(s): {sorted(invalid)}"
            )

        self.concept_embedding = EHRtokenEmbedding(
            self.itemid_size, self.hidden_size
        )

        if "time" not in self.ablation:
            self.time_embedding = TimeEmbedding(1, self.hidden_size)

        if "value" not in self.ablation:
            if _get_value_embedding_type(args) == "simple":
                self.value_embedding = ValueEmbedding(self.hidden_size)
            else:
                self.value_embedding = ContinuousValueEmbedding(
                    self.hidden_size
                )

        if "unit" not in self.ablation:
            self.unit_embedding = UnitEmbedding(
                self.unit_size, self.hidden_size
            )

        self.age_embedding = AgeEmbedding(
            self.max_age, self.hidden_size
        )
        self.gender_embedding = GenderEmbedding(
            self.gender_size, self.hidden_size
        )
        self.task_embedding = TaskEmbedding(
            self.task_size, self.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        value_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        time_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        age_ids: torch.Tensor,
        gender_ids: torch.Tensor,
        task_ids: torch.Tensor,
        inputs_embeds=None,
        **legacy_kwargs,
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        event_embeddings = self.concept_embedding(input_ids)

        if "time" not in self.ablation:
            event_embeddings = event_embeddings + self.time_embedding(time_ids)

        if "value" not in self.ablation:
            event_embeddings = event_embeddings + self.value_embedding(value_ids)

        if "unit" not in self.ablation:
            event_embeddings = event_embeddings + self.unit_embedding(unit_ids)

        age_embed = self.age_embedding(age_ids)
        gender_embed = self.gender_embedding(gender_ids)
        task_embed = self.task_embedding(task_ids)

        return torch.cat(
            (task_embed, age_embed, gender_embed, event_embeddings),
            dim=1,
        )
