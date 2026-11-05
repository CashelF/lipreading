from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from backend.preview_decoding import greedy_preview,PreviewTimeout,merged_preview_model


def test_merging_copy_cannot_change_trusted_parameters_or_registry():
    class Lora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([1.]))
        def merge_and_unload(self):
            with torch.no_grad():self.weight.add_(2.)
            return self
    original=torch.nn.Module()
    original.llm=Lora()
    original.encoder=torch.nn.Linear(1,1)
    original.use_lora=True
    candidate=merged_preview_model(original)
    assert original.llm.weight.item()==1.
    assert candidate.llm.weight.item()==3.
    assert candidate.llm is not original.llm
    assert candidate.encoder is original.encoder
    assert original.use_lora and not candidate.use_lora

class Qwen:
    def __call__(self,inputs_embeds=None,input_ids=None,past_key_values=None,**kwargs):
        history=[] if past_key_values is None else past_key_values[0][0].flatten().tolist()
        values=[0]*inputs_embeds.shape[1] if inputs_embeds is not None else input_ids[0].tolist()
        logits=[]
        for value in values:
            history.append(value)
            next_token=7 if len(history)>=11 else (int(sum(history))+len(history))%6+1
            row=torch.full((8,),-10.);row[next_token]=10.;logits.append(row)
        cache=torch.tensor(history).reshape(1,1,-1,1)
        return SimpleNamespace(last_hidden_state=torch.stack(logits)[None],past_key_values=((cache,cache.clone()),))

def model():return SimpleNamespace(use_lora=False,llm=SimpleNamespace(model=Qwen(),lm_head=lambda x:x),llm_tokenizer=SimpleNamespace(eos_token_id=7))

@pytest.mark.parametrize('kind',['correct','wrong','short','empty'])
def test_speculative_accept_reject_and_cache_crop_match_greedy(kind):
    host=model();prefix=torch.zeros((1,3,8))
    baseline,base_stats=greedy_preview(host,prefix,block_size=0)
    draft={'correct':baseline.tokens.tolist(),'wrong':[1]*20,'short':baseline.tokens.tolist()[:4],'empty':[]}[kind]
    actual,stats=greedy_preview(host,prefix,draft=draft)
    assert torch.equal(actual.tokens,baseline.tokens)
    assert actual.ended
    if kind=='correct':
        assert stats['draft_accepted']>0
        assert stats['forward_calls']<base_stats['forward_calls']

def test_preview_generation_is_bounded_and_deadline_is_enforced():
    prefix=torch.zeros((1,3,8))
    candidate,stats=greedy_preview(model(),prefix,draft=[1]*100,max_tokens=3)
    assert candidate.tokens.numel()==3 and not candidate.ended
    with pytest.raises(PreviewTimeout):greedy_preview(model(),prefix,deadline=0)
