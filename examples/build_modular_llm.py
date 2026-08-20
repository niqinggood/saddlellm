"""Build the same kind of modular network directly in Python."""

from saddlellm import SaddleModelBuilder


builder = (
    SaddleModelBuilder(
        "saddle-hybrid-demo",
        family="saddle",
        hidden_size=512,
        vocab_size=32000,
        max_position_embeddings=4096,
    )
    .defaults(
        attention={
            "preset": "gqa",
            "backend": "sdpa",
            "num_heads": 8,
            "num_kv_heads": 2,
        },
        ffn={"preset": "swiglu", "intermediate_size": 1536},
        residual={"preset": "depth_scaled"},
    )
    .add_layers(2, name="dense-stem")
    .add_layer(
        name="parallel-dense",
        attention={"kind": "mqa"},
        residual={"topology": "parallel"},
    )
    .add_layer(
        name="sparse-tail",
        attention={
            "kind": "mla",
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "mla_cache_mode": "latent",
        },
        ffn={
            "kind": "moe",
            "intermediate_size": 1536,
            "num_experts": 8,
            "experts_per_token": 2,
            "expert_intermediate_size": 768,
            "shared_expert": True,
        },
        residual={
            "topology": "serial",
            "learnable": True,
            "attention_scale": 0.5,
            "ffn_scale": 0.5,
            "initialization": "depth_scaled",
        },
    )
)

blueprint = builder.build_blueprint()  # serializable architecture source
model = builder.build()                # torch.nn.Module ready for training

print(blueprint.to_dict()["analysis"])
print([layer.config.attention_kind for layer in model.layers])
print([layer.config.ffn_kind for layer in model.layers])
