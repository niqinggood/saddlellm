"""Independent numerical and real-training checks for the public distiller."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from saddlellm.distillation.DistillationTrainer import DistillationTrainer


def _tokenizer(swapped=False):
    tokens = ['<unk>', '<bos>', '<eos>', 'user', 'assistant', ':', 'one', 'two', 'three', 'four', 'five', 'six']
    vocab = dict(zip(tokens, range(len(tokens))))
    if swapped:
        vocab['one'], vocab['two'] = vocab['two'], vocab['one']
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<bos>',
                                       eos_token='<eos>', pad_token='<eos>')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return tokenizer


def _model(hidden=8, vocab=12):
    return GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_embd=hidden, n_head=2, n_layer=1,
                                    n_positions=32, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                    bos_token_id=1, eos_token_id=2, pad_token_id=2))


def _distiller(**kwargs):
    return DistillationTrainer(_model(), _model(), tokenizer=_tokenizer(), teacher_tokenizer=_tokenizer(),
                               device_map='cpu', alpha_hidden=kwargs.pop('alpha_hidden', 0), **kwargs)


def _loss_only(kd=0.5, hidden=0, temperature=2):
    trainer = object.__new__(DistillationTrainer)
    trainer.alpha_ce, trainer.alpha_hidden, trainer.temperature = kd, hidden, temperature
    return trainer


def _log_softmax(values):
    denominator = sum(math.exp(x) for x in values)
    return [x - math.log(denominator) for x in values]


def test_public_alias_preserves_class_api():
    from saddlellm import distill_model

    assert distill_model is DistillationTrainer


def test_loss_matches_hand_calculated_shifted_token_mean_and_gradients():
    student = torch.tensor([[[9., -9., 1.], [1., 2., -1.], [-2., 1., 3.], [8., 0., -8.], [3., 4., 5.]]], requires_grad=True)
    teacher = torch.tensor([[[0., 2., 1.], [2., 0., 1.], [1., 3., -1.], [7., 2., 4.], [5., 0., 8.]]], requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 1, -100]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    trainer = _loss_only(kd=.4, temperature=2)
    actual = trainer.compute_distillation_loss(SimpleNamespace(logits=student), SimpleNamespace(logits=teacher), labels, attention)
    ce, kl = [], []
    for position, target in [(1, 2), (2, 1)]:
        s = student.detach()[0, position].tolist()
        t = teacher.detach()[0, position].tolist()
        ce.append(-_log_softmax(s)[target])
        sl, tl = _log_softmax([x / 2 for x in s]), _log_softmax([x / 2 for x in t])
        kl.append(4 * sum(math.exp(q) * (q - p) for p, q in zip(sl, tl)))
    assert float(actual) == pytest.approx(.6 * sum(ce) / 2 + .4 * sum(kl) / 2, abs=1e-6)
    actual.backward()
    assert teacher.grad is None
    assert student.grad[0, [0, 3, 4]].count_nonzero() == 0
    assert student.grad[0, 1:3].abs().sum() > 0
    assert trainer._last_tokens == 2


def test_left_padding_is_ignored_but_real_eos_is_supervised():
    logits = torch.randn(1, 5, 3, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    labels = torch.tensor([[2, 0, -100, 1, 2]])
    loss = _loss_only().compute_distillation_loss(SimpleNamespace(logits=logits),
        SimpleNamespace(logits=teacher), labels, torch.tensor([[0, 1, 1, 1, 1]]))
    loss.backward()
    assert logits.grad[0, :2].count_nonzero() == 0
    assert logits.grad[0, 3].abs().sum() > 0  # predicts real EOS, also the pad id
    assert logits.grad[0, 4].count_nonzero() == 0
    assert teacher.grad is None


def test_hidden_matching_uses_same_predictor_mask_and_detaches_teacher():
    s = torch.randn(1, 4, 3, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    sh = torch.tensor([[[99., 99.], [1., 3.], [99., 99.], [99., 99.]]], requires_grad=True)
    th = torch.zeros_like(sh, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, -100]])
    trainer = _loss_only(kd=0, hidden=1)
    loss = trainer.compute_distillation_loss(SimpleNamespace(logits=s, hidden_states=(sh,)),
        SimpleNamespace(logits=t, hidden_states=(th,)), labels)
    assert float(loss) == pytest.approx(5)
    loss.backward()
    assert th.grad is None and t.grad is None
    assert sh.grad[0, [0, 2, 3]].count_nonzero() == 0
    assert sh.grad[0, 1].tolist() == [1., 3.]


@pytest.mark.parametrize('kwargs', [{'temperature': 0}, {'temperature': float('nan')}, {'alpha_ce': .8, 'alpha_hidden': .3},
                                    {'alpha_ce': -1}, {'alpha_hidden': True}, {'max_length': 1}])
def test_invalid_loss_controls_fail_before_model_loading(kwargs):
    with pytest.raises(ValueError):
        DistillationTrainer('must-not-load', 'must-not-load', **kwargs)


def test_heterogeneous_token_ids_hidden_dimensions_and_shared_teacher_are_rejected():
    with pytest.raises(ValueError, match='vocabularies'):
        DistillationTrainer(_model(), _model(), tokenizer=_tokenizer(), teacher_tokenizer=_tokenizer(True), device_map='cpu')
    with pytest.raises(ValueError, match='hidden dimensions'):
        DistillationTrainer(_model(12), _model(8), tokenizer=_tokenizer(), teacher_tokenizer=_tokenizer(), device_map='cpu')
    model = _model()
    with pytest.raises(ValueError, match='independent'):
        DistillationTrainer(model, model, tokenizer=_tokenizer(), teacher_tokenizer=_tokenizer(), device_map='cpu')
    assert all(p.requires_grad for p in model.parameters())


def test_hidden_and_vocab_shape_mismatches_fail_in_direct_loss():
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ValueError, match='vocabulary'):
        _loss_only().compute_distillation_loss(SimpleNamespace(logits=torch.randn(1, 3, 4)),
            SimpleNamespace(logits=torch.randn(1, 3, 5)), labels)
    with pytest.raises(ValueError, match='hidden-state dimensions'):
        _loss_only(hidden=.3).compute_distillation_loss(SimpleNamespace(logits=torch.randn(1, 3, 4), hidden_states=(torch.randn(1, 3, 2),)),
            SimpleNamespace(logits=torch.randn(1, 3, 4), hidden_states=(torch.randn(1, 3, 3),)), labels)


def test_prompt_mask_is_configurable_and_explicit_labels_are_kept():
    trainer = _distiller()
    record = {'prompt': 'one two', 'completion': 'three four'}
    masked = trainer._prepare([record])[0]
    assert -100 in masked['labels'] and any(x >= 0 for x in masked['labels'][1:])
    trainer.mask_prompt = False
    unmasked = trainer._prepare([record])[0]
    assert unmasked['labels'] == unmasked['input_ids']
    explicit = {'input_ids': [1, 6, 7, 2], 'labels': [-100, -100, 7, 2]}
    assert trainer._prepare([explicit])[0]['labels'] == explicit['labels']


def test_real_batched_fit_evaluates_heldout_freezes_teacher_and_reloads(tmp_path):
    torch.manual_seed(17)
    trainer = _distiller(alpha_hidden=.2)
    teacher_before = copy.deepcopy(trainer.teacher.state_dict())
    student_before = copy.deepcopy(trainer.student.state_dict())
    train = [{'input_ids': row, 'labels': [-100, -100] + row[2:]} for row in
             [[1, 6, 7, 2], [1, 7, 8, 9, 2], [1, 8, 6, 2]]]
    validation = [{'input_ids': row} for row in [[1, 10, 11, 2], [1, 11, 10, 6, 2]]]
    batches, validation_forwards = [], []
    def trace(model, args, kwargs):
        if torch.is_grad_enabled():
            batches.append(kwargs['input_ids'].shape[0])
        else:
            validation_forwards.append(kwargs['input_ids'].detach().tolist())
    hook = trainer.student.register_forward_pre_hook(trace, with_kwargs=True)
    output = tmp_path / 'student'
    assert trainer.fit(train, validation, epochs=2, batch_size=2, learning_rate=.003,
                       output_dir=str(output), logging_steps=100) is trainer
    hook.remove()
    assert batches == [2, 1, 2, 1]
    assert len(validation_forwards) == 3
    assert trainer.metrics_['steps'] == 4
    assert trainer.metrics_['eval_samples'] == 2 and trainer.metrics_['eval_tokens'] == 7
    assert trainer.metrics_['validation_before']['eval_samples'] == 2
    assert any(not torch.equal(student_before[k], v) for k, v in trainer.student.state_dict().items())
    assert all(torch.equal(teacher_before[k], v) for k, v in trainer.teacher.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in trainer.teacher.parameters())
    first = trainer.evaluate(validation, batch_size=1)
    second = trainer.evaluate(validation, batch_size=2)
    assert first['eval_loss'] == pytest.approx(second['eval_loss'], abs=1e-6)
    assert json.loads((output / 'distillation_metrics.json').read_text())['metrics']['steps'] == 4
    restored = DistillationTrainer.load(str(output), device_map='cpu', local_files_only=True)
    assert restored.teacher is None
    assert restored.student.config.vocab_size == 12
    trainer.student.eval()
    ids = torch.tensor([[1, 6, 7]])
    with torch.no_grad():
        assert torch.allclose(restored.student(ids).logits, trainer.student(ids).logits, atol=1e-6)
    with pytest.raises(ValueError, match='requires a teacher'):
        restored.fit(train)


def test_validation_leakage_and_no_targets_rejected_without_output(tmp_path):
    trainer = _distiller(max_length=3)
    with pytest.raises(ValueError, match='overlap'):
        trainer.fit([{'input_ids': [1, 6, 7, 8]}], [{'input_ids': [1, 6, 7, 9]}], output_dir=str(tmp_path / 'invalid'))
    assert not (tmp_path / 'invalid').exists()
    with pytest.raises(ValueError, match='no supervised'):
        trainer.fit([{'input_ids': [1, 6], 'labels': [-100, -100]}], output_dir=str(tmp_path / 'invalid'))


def test_distillation_blocks_shared_source_cases_and_writes_bound_audit(tmp_path):
    trainer = _distiller()
    train = [{'case_id': 'same-case', 'text': 'one two three'}]
    heldout = [{'case_id': 'same-case', 'text': 'four five six'}]
    output = tmp_path / 'audit'
    with pytest.raises(ValueError, match='leakage'):
        trainer.fit(train, heldout, output_dir=str(output))
    report = json.loads((output / 'split_audit.json').read_text())
    assert report['passed'] is False
    assert report['source_overlap'] > 0
    assert 'same-case' not in json.dumps(report)
    assert not (output / 'model.safetensors').exists()


def test_native_public_distillation_trains_and_reloads_with_chunked_loss(tmp_path):
    from saddlellm.models.ModelLoader import load_causal_lm
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    config = SaddleModelConfig(vocab_size=12, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, attention_backend='eager', loss_chunk_size=2)
    student = SaddleForCausalLM(config)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.lm_head.weight.add_(.1)
    before = copy.deepcopy(student.state_dict())
    trainer = DistillationTrainer(teacher, student, tokenizer=_tokenizer(),
        teacher_tokenizer=_tokenizer(), device_map='cpu', alpha_hidden=0)
    output = tmp_path / 'native'
    trainer.fit([{'input_ids': [1, 6, 7, 2]}], [{'input_ids': [1, 10, 11, 2]}],
        epochs=1, batch_size=2, output_dir=str(output), logging_steps=100)
    assert trainer.metrics_['steps'] == 1
    assert trainer.metrics_['split_audit']['passed']
    assert any(not torch.equal(before[k], v) for k, v in student.state_dict().items())
    restored = load_causal_lm(str(output), device='cpu', dtype=torch.float32, local_files_only=True)
    student.eval()
    ids = torch.tensor([[1, 6, 7]])
    with torch.no_grad():
        assert torch.allclose(student(ids).logits, restored(ids).logits, atol=1e-6)
