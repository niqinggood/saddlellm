"""Preference labels must remain on responses after truncating long prompts."""
import pytest

from saddlellm.training.NativePostTraining import _preference_features


class CharacterTokenizer:
    def __init__(self, trailing_eos=False):
        self.trailing_eos=trailing_eos

    def __call__(self, text, *, add_special_tokens=True, truncation=False, max_length=None):
        ids=[ord(c) for c in text]
        if add_special_tokens and self.trailing_eos:
            ids=[1]+ids+[2]
        if truncation:
            ids=ids[:max_length]
        return {'input_ids':ids}


@pytest.mark.parametrize('max_length,max_prompt_length',[(12,4),(8,4),(8,30)])
@pytest.mark.parametrize('special_tokens',[False,True])
def test_long_prompt_is_actually_truncated_and_answers_survive(max_length,max_prompt_length,special_tokens):
    row=_preference_features(dict(prompt='abcdefghijklmnop',chosen='YES',rejected='NO'),
        CharacterTokenizer(special_tokens),max_length,max_prompt_length)
    assert row is not None
    contexts=[]
    for name,answer in [('chosen','YES'),('rejected','NO')]:
        ids,labels=row[name+'_input_ids'],row[name+'_labels']
        boundary=next(i for i,label in enumerate(labels) if label!=-100)
        assert 0<boundary<=min(max_prompt_length,max_length-1)
        assert len(ids)<=max_length
        contexts.append(ids[:boundary])
        # The separator/answer can be shortened to the available capacity;
        # no prompt character may be labeled as a preferred/rejected answer.
        supervised=[label for label in labels if label!=-100]
        assert all(label in [ord(c) for c in ' '+answer]+[2] for label in supervised)
        assert any(label in [ord(c) for c in answer] for label in supervised)
    assert contexts[0]==contexts[1]


def test_short_prompt_retains_joint_tokenization_and_exact_response_mask():
    row=_preference_features(dict(prompt='question',chosen='YES',rejected='NO'),CharacterTokenizer(),32,16)
    assert row['chosen_input_ids']==list(map(ord,'question YES'))
    assert row['chosen_labels']==[-100]*9+list(map(ord,'YES'))
    assert row['rejected_labels']==[-100]*9+list(map(ord,'NO'))


@pytest.mark.parametrize('answer',['',' ','\n\t'])
def test_empty_response_cannot_be_replaced_with_a_prompt_label(answer):
    assert _preference_features(dict(prompt='long question',chosen=answer,rejected='NO'),
        CharacterTokenizer(),8,8) is None


@pytest.mark.parametrize('limits',[(1,1),(0,1),(10,0),(10,-1),(True,3),(10,True),(10,2.5)])
def test_invalid_lengths_fail_before_tokenization(limits):
    with pytest.raises(ValueError,match='length'):
        _preference_features(dict(prompt='q',chosen='YES',rejected='NO'),CharacterTokenizer(),*limits)


def test_only_response_tokens_survive_causal_shift():
    row=_preference_features(dict(prompt='abcdefghijk',chosen='YES',rejected='NO'),CharacterTokenizer(),6,3)
    for name in ('chosen','rejected'):
        assert row[name+'_labels'][0]==-100
        assert any(label!=-100 for label in row[name+'_labels'][1:])


def test_boundary_merge_is_supervised_and_shared_context_stays_equal():
    class BoundaryTokenizer(CharacterTokenizer):
        def __call__(self,text,**kwargs):
            result=super().__call__(text,**kwargs)
            if text=='q YES':result['input_ids']=[999,ord('E'),ord('S')]
            elif text=='q NO':result['input_ids']=[998,ord('O')]
            return result
    # With the entire one-token prompt merged into the answer there is no
    # shared causal context; do not quietly train two unconditional strings.
    assert _preference_features(dict(prompt='q',chosen='YES',rejected='NO'),BoundaryTokenizer(),8,4) is None
