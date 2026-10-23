import json

import torch


def test_real_lora_training_preserves_padded_vocabulary_and_reloadable_adapter(tmp_path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    from saddlellm.agents import RealWorldOperators as operators
    from saddlellm.training.PeftDispatch import unquantized_lora_dispatch

    config = Qwen3Config(vocab_size=24, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=8, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config)
    base = tmp_path / 'base'; model.save_pretrained(base)
    model.config._name_or_path = str(base)

    class Tokenizer:
        def __len__(self): return 20
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[1, 2, 3, 4] for _ in texts])
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
        def save_pretrained(self, path):
            (path / 'tokenizer_config.json').write_text('{}')

    monkeypatch.setattr(operators, '_load_tokenizer', lambda *a: Tokenizer())
    monkeypatch.setattr(operators, '_load_model', lambda *a, **k: model)
    monkeypatch.setattr(operators, '_data_config', lambda *a: (['training'], ['validation'], {}))
    output = tmp_path / 'sft'; output.mkdir()
    result = operators._sft_operator({'model': str(base), 'device': 'cpu', 'maxSteps': 1,
          'batchSize': 1, 'maxLength': 8, 'targetModules': ['q_proj'], 'loraR': 2}, None, output)
    assert result['status'] == 'success'
    assert json.loads((output / 'merged_model/config.json').read_text())['vocab_size'] == 24
    weights = load_file(output / 'lora_adapter/adapter_model.safetensors')
    assert weights and all('lora_' in key for key in weights)
    base_model = Qwen3ForCausalLM.from_pretrained(base)
    with unquantized_lora_dispatch(base_model):
        reloaded = PeftModel.from_pretrained(base_model, output / 'lora_adapter')
    assert reloaded.get_input_embeddings().num_embeddings == 24
    assert torch.isfinite(reloaded(input_ids=torch.tensor([[1, 2, 3]])).logits).all()
