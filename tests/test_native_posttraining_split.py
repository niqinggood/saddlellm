"""Validation must hold out whole source conversations and leave training data."""
import pytest

from saddlellm.training.NativePostTraining import _split_features


def test_all_turns_of_a_conversation_stay_on_one_side():
    groups=[10,10,20,20,20,30,40,40]
    features=[{'input_ids':[group,index],'labels':[-100,index]} for index,group in enumerate(groups)]
    for seed in (1,42,2026):
        train,evaluation=_split_features(features,.5,seed,groups=groups)
        train_groups={item['input_ids'][0] for item in train}
        eval_groups={item['input_ids'][0] for item in evaluation}
        assert len(train_groups)==len(eval_groups)==2
        assert train_groups.isdisjoint(eval_groups)
        assert len(train)+len(evaluation)==len(features)
        again=_split_features(features,.5,seed,groups=groups)
        assert train.features==again[0].features and evaluation.features==again[1].features


def test_many_turns_from_one_conversation_are_not_an_independent_holdout():
    with pytest.raises(ValueError,match='source records'):
        _split_features([{'input_ids':[i]} for i in range(8)],.2,42,groups=[0]*8)


def test_high_fraction_on_tiny_dataset_keeps_a_training_record():
    train,evaluation=_split_features([{'input_ids':[0]},{'input_ids':[1]}],.99,42)
    assert len(train)==len(evaluation)==1


@pytest.mark.parametrize('fraction',[-.1,1,1.5,float('nan'),float('inf'),True])
def test_invalid_split_does_not_silently_disable_validation(fraction):
    with pytest.raises(ValueError,match='validation_split'):
        _split_features([{'input_ids':[0]},{'input_ids':[1]}],fraction,42)


def test_native_sft_passes_source_groups_to_split(monkeypatch,tmp_path):
    import json
    from saddlellm.training import NativePostTraining as native

    source=tmp_path/'records.jsonl'
    source.write_text('\n'.join(json.dumps({'id':i}) for i in range(3)),encoding='utf-8')
    class Tokenizer:
        pad_token_id=0
        eos_token_id=1
    monkeypatch.setattr(native,'_activate_local_device',lambda:None)
    monkeypatch.setattr(native.SaddleModelAdapter,'load_tokenizer',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(native,'_sft_features',lambda record,*a:[{'input_ids':[record['id']]}]*(record['id']+1))
    class SplitReached(Exception):pass
    def capture(features,fraction,seed,*,groups):
        assert groups==[0,1,1,2,2,2]
        assert len(features)==6
        raise SplitReached
    monkeypatch.setattr(native,'_split_features',capture)
    with pytest.raises(SplitReached):
        native.train_native_sft(native.NativeSFTConfig(model_path='unused',dataset_path=str(source),
            output_path=str(tmp_path/'output'),validation_split=.3))
