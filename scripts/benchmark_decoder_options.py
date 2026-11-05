#!/usr/bin/env python3
"""Isolated decoder experiments on local examples; save numeric results only."""
import argparse,copy,json,logging,statistics,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_speaking_gate import load_records,packets_for_record,write_private
from scripts.benchmark_latency import gpu_sample
from backend.stream import ClipBuffer,resample
from backend.model import FlareModel
from backend.grounding import capture_candidate,verify_candidate,visual_prefix,qwen_parts
from backend.preview_decoding import greedy_preview

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['merge','compile','graphs','speculative'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
 if args.repeats<1:raise ValueError('repeats must be positive')
 with urllib.request.urlopen('http://127.0.0.1:8000/api/status') as response:
  if json.load(response)['busy']:raise SystemExit('User session active')
 import torch
 runtime=FlareModel();runtime.load();reference=runtime.model;candidate=reference
 if args.mode in ('merge','compile','graphs'):
  candidate=copy.copy(reference);candidate._modules=reference._modules.copy()
  candidate.llm=copy.deepcopy(reference.llm)
  if args.mode=='merge':
   candidate.llm=candidate.llm.merge_and_unload();candidate.use_lora=False
  elif args.mode=='graphs':
   from backend.decoder_graphs import DecoderGraphs
   qwen,_=qwen_parts(candidate)
   qwen.forward=DecoderGraphs(qwen.forward)
  else:
   qwen,_=qwen_parts(candidate);original=qwen.forward
   compiled=torch.compile(original, dynamic=True, mode='reduce-overhead')
   def forward(*a,**kw):
    if kw.get('input_ids') is not None and kw['input_ids'].shape[1]==1:return compiled(*a,**kw)
    return original(*a,**kw)
   qwen.forward=forward
 report={'mode':args.mode,'caption_text_saved':False,'repeats':args.repeats,'limitations':['Development examples without reference words; equality is not WER.','Speculation compared with greedy target, separately from beam 30.'],'rows':[]}
 def beam(model,sample):
  with capture_candidate(runtime._search) as captured:
   output,_=runtime._search([model],sample,beam_size=30,audio_only=False,video_only=True,temp=1.2)
  return output[0],captured[0]
 with torch.inference_mode():
  for directory,manifest in load_records().values():
   if manifest['take']!=2:continue
   with urllib.request.urlopen('http://127.0.0.1:8000/api/status') as response:
    if json.load(response)['busy']:raise RuntimeError('User session active')
   packets=packets_for_record(directory,manifest);draft=[]
   for end in (1200,2000,4000):
    buffer=ClipBuffer()
    for stamp,packet in packets:
     if stamp>end:break
     buffer.append(packet)
    clip=buffer.finish();frames,_=resample(clip);prepared=runtime.preprocessor(frames,25);sample=gpu_sample(prepared)
    print(json.dumps({'starting':manifest['label'],'end_ms':end}),flush=True)
    if args.mode=='speculative':
     prefix=visual_prefix(reference,sample)
     trials={'before':[],'after':[]};same=True
     for trial in range(args.repeats+1):
      outputs={}
      for name in (('before','after') if trial%2 else ('after','before')):
       torch.cuda.synchronize();t=time.perf_counter()
       result,metrics=greedy_preview(reference,prefix,block_size=0) if name=='before' else greedy_preview(reference,prefix,draft=draft)
       torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
       if trial:trials[name].append(elapsed)
       outputs[name]=(result,metrics)
      base,base_metrics=outputs['before'];new,new_metrics=outputs['after'];same &= torch.equal(base.tokens,new.tokens)
     before=statistics.median(trials['before']);after=statistics.median(trials['after'])
     text_equal=torch.equal(base.tokens,new.tokens);draft=new.tokens.tolist()
     _,beam_candidate=beam(reference,sample)
     row={'label':manifest['label'],'end_ms':end,'before_ms':before,'after_ms':after,'tokens_equal':same,'matches_beam30':torch.equal(base.tokens,beam_candidate.tokens),'baseline':base_metrics,'candidate':new_metrics,'trials_ms':trials}
    else:
     values={};trials={'before':[],'after':[]};verified={'before':[],'after':[]};equal=True;full_equal=True;warmup={}
     for trial in range(args.repeats+1):
      order=[('before',reference),('after',candidate)]
      if trial%2:order.reverse()
      for name,model in order:
       torch.cuda.synchronize();t=time.perf_counter();text,selected=beam(model,sample);torch.cuda.synchronize();decode_ms=(time.perf_counter()-t)*1000
       t=time.perf_counter();verification=verify_candidate(model,sample,selected);torch.cuda.synchronize();verification_ms=(time.perf_counter()-t)*1000
       values[name]=(text,selected,verification,decode_ms,verification_ms)
       if trial:trials[name].append(decode_ms);verified[name].append(verification_ms)
       else:
        warmup[name]=decode_ms
        print(json.dumps({'warmup':name,'ms':decode_ms}),flush=True)
      equal &= torch.equal(values['before'][1].tokens,values['after'][1].tokens)
      left,right=values['before'][1].alternatives,values['after'][1].alternatives
      full_equal &= len(left)==len(right) and all(torch.equal(x.tokens,y.tokens) and x.beam_score==y.beam_score for x,y in zip(left,right))
     a,b=values['before'],values['after']
     row={'label':manifest['label'],'end_ms':end,'before_ms':statistics.median(trials['before']),'after_ms':statistics.median(trials['after']),'before_verification_ms':statistics.median(verified['before']),'after_verification_ms':statistics.median(verified['after']),'text_equal':a[0]==b[0],'tokens_equal':equal,'verification_equal':a[2]['accepted']==b[2]['accepted'],'before_accepted':a[2]['accepted'],'after_accepted':b[2]['accepted'],'trials_ms':trials}
     row.update(full_beam_equal=full_equal,warmup_ms=warmup)
     if args.mode=='graphs':row.update(graph_hits=qwen.forward.hits,graph_misses=qwen.forward.misses)
    report['rows'].append(row);write_private(args.output,report);print(json.dumps(row),flush=True)
 report['complete']=True;write_private(args.output,report)
if __name__=='__main__':main()
