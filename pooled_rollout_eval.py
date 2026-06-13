#!/usr/bin/env python
"""
Phase D: OOD rollout suite (IT, SB, RP, AIL) — extract (GPU, CACHED) + score.

Extracts pre + response-token activations for the four Apollo-style rollout sets,
CACHES them as acts_OOD_<KEY>_{layer,pre}_*.npy + labels_OOD_<KEY>.npy (reused on
re-run — no GPU needed next time), then scores P_S/P_C/POOLED/I'/Z(_mean) under
honest seeds + DiffMean twin, with de-inversion (report max(auc,1-auc) + inverted
flag) and balAcc@1%FPR.

Conventions: RP/AIL/SB single-turn (resp = last content token of assistant; clean
for RP/AIL). IT multi-turn -> resp falls back to prompt-end (flagged approx). Our
probes are response-trained -> read the resp column; pre column is a position
cross-check.

Env: HF_TOKEN (only if any set uncached), OUTPUT_DIR (~/output),
ROLLOUT_DIR (dir with *_rollouts*.json; default ~/Truth_is_Universal/output).
"""
import os, sys, gc, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_utils import multiseed_best_layer, fit_diffmean, _bootstrap_auc
import ood_eval_utils as ood

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
ROLLOUT_DIR = os.environ.get('ROLLOUT_DIR', os.path.expanduser('~/Truth_is_Universal/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, TEST_SIZE, MAX_ITER = 32, 10, 1.0, 0.2, 2000
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
PS = pickle.load(open(op / 'probe_state.pkl', 'rb')); g = globals(); g.update(PS)

SETS = [('IT', ood.load_insider_trading_rollouts, True),   # multi-turn -> resp approx
        ('SB', ood.load_sandbagging_rollouts, False),
        ('RP', ood.load_roleplaying_rollouts, False),
        ('AIL', ood.load_ai_liar_rollouts, False)]

def _cached(key):
    return ((op / f'acts_OOD_{key}_layer_000.npy').exists()
            and (op / f'acts_OOD_{key}_pre_layer_000.npy').exists()
            and (op / f'labels_OOD_{key}.npy').exists())

data = {}
for key, loader, _multi in SETS:
    try:
        rolls, labs = loader(ROLLOUT_DIR)
    except Exception as e:
        print(f'[{key}] load failed: {e}'); continue
    if len(labs) and len(np.unique(labs)) >= 2:
        data[key] = (rolls, labs)

need_model = any(not _cached(k) for k in data)
feats, feats_pre, labels = {}, {}, {}

if need_model:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from tqdm.auto import tqdm
    hf = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=hf)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf, device_map='auto',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model.eval(); print('Model loaded.')
    _eot = set([tok.eos_token_id])
    for t in ['<|eot_id|>', '<|end_of_text|>']:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0: _eot.add(i)
    def ct(msgs, addgen):
        o = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=addgen, return_tensors='pt')
        if hasattr(o, 'input_ids'): o = o.input_ids
        elif isinstance(o, dict): o = o['input_ids']
        if hasattr(o, 'tolist'): o = o.tolist()
        while isinstance(o, list) and o and isinstance(o[0], list): o = o[0]
        return [int(x) for x in o]
    def resp_ids(tr):
        if tr and tr[-1]['role'] == 'assistant':
            ids = ct(tr, False)
            while len(ids) > 1 and ids[-1] in _eot: ids = ids[:-1]
            return ids
        return ct(tr, True)
    def pre_ids(tr):
        m = [x for x in tr]
        if m and m[-1]['role'] == 'assistant': m = m[:-1]
        return ct(m if m else tr, True)
    def extract(id_lists, tag):
        import torch
        out = {}
        for s in tqdm(range(0, len(id_lists), 4), desc=tag):
            batch = id_lists[s:s+4]; ml = max(len(x) for x in batch)
            ids = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids); pos = []
            for i, x in enumerate(batch):
                ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1; pos.append(len(x)-1)
            ids, mask = ids.to(model.device), mask.to(model.device)
            pt = torch.tensor(pos, device=model.device)
            with torch.no_grad():
                hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
            for li, lh in enumerate(hs):
                bi = torch.arange(lh.shape[0], device=lh.device)
                out.setdefault(li, []).extend(list(lh[bi, pt, :].float().cpu().numpy()))
        return {li: np.stack(v) for li, v in out.items()}

for key, (rolls, labs) in data.items():
    if _cached(key):
        print(f'[{key}] cached');
        feats[key] = {li: np.load(op / f'acts_OOD_{key}_layer_{li:03d}.npy') for li in range(N_LAYERS) if (op/f'acts_OOD_{key}_layer_{li:03d}.npy').exists()}
        feats_pre[key] = {li: np.load(op / f'acts_OOD_{key}_pre_layer_{li:03d}.npy') for li in range(N_LAYERS) if (op/f'acts_OOD_{key}_pre_layer_{li:03d}.npy').exists()}
        labels[key] = np.load(op / f'labels_OOD_{key}.npy'); continue
    print(f'[{key}] extracting {len(rolls)} rollouts (resp+pre)...')
    rids = [resp_ids(r['transcript']) for r in rolls]
    pids = [pre_ids(r['transcript']) for r in rolls]
    fr = extract(rids, f'{key}_resp'); fp = extract(pids, f'{key}_pre')
    for li in fr: np.save(op / f'acts_OOD_{key}_layer_{li:03d}.npy', fr[li])
    for li in fp: np.save(op / f'acts_OOD_{key}_pre_layer_{li:03d}.npy', fp[li])
    np.save(op / f'labels_OOD_{key}.npy', labs)
    feats[key], feats_pre[key], labels[key] = fr, fp, labs
    print(f'  [{key}] cached acts_OOD_{key}_*')

if need_model:
    import torch
    del model; torch.cuda.empty_cache(); gc.collect(); print('Model freed.')

# ── build resp probes (retrain) + POOLED ─────────────────────────────────────
def load(tag, pos): return {li: np.load(op/f'acts_{tag}_{pos}_layer_{li:03d}.npy') for li in range(N_LAYERS) if (op/f'acts_{tag}_{pos}_layer_{li:03d}.npy').exists()}
fr_ = {'P_S': load('P_S','resp'), 'P_C': load('P_C','resp'), 'I_S': load('I_S','resp'), 'I_C': load('I_C','resp')}
Yd = {'P_S':labels_P_S_lied,'P_C':labels_P_C_lied,'I_S':labels_I_S,'I_C':labels_I_C}
Md = {'P_S':mask_P_S,'P_C':mask_P_C,'I_S':mask_I_S,'I_C':mask_I_C}
Gd = {'P_S':groups_P_S,'P_C':groups_P_C,'I_S':groups_I_S,'I_C':groups_I_C}
BL = {}
for t in fr_: BL[t],_ = multiseed_best_layer(fr_[t],Yd[t],Md[t],n_seeds=N_SEEDS,C=PROBE_C,groups=Gd[t])
pl = sorted(set(fr_['P_S'])&set(fr_['P_C']))
fr_['POOLED']={li:np.concatenate([fr_['P_S'][li][mask_P_S],fr_['P_C'][li][mask_P_C]],0) for li in pl}
Yd['POOLED']=np.concatenate([labels_P_S_lied[mask_P_S],labels_P_C_lied[mask_P_C]])
Gd['POOLED']=np.concatenate([np.array([f'S:{x}' for x in groups_P_S[mask_P_S]]),np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])])
Md['POOLED']=np.ones(len(Yd['POOLED']),bool)
BL['POOLED'],_=multiseed_best_layer(fr_['POOLED'],Yd['POOLED'],Md['POOLED'],n_seeds=N_SEEDS,C=PROBE_C,groups=Gd['POOLED'])

def Xy(t): return {li:fr_[t][li][Md[t]] for li in fr_[t]}, Yd[t][Md[t]], (np.asarray(Gd[t])[Md[t]] if t!='POOLED' else Gd['POOLED'])
def score(t, tf, ty):
    X,y,gr = Xy(t); bl=BL[t]; aucs=[]; last=None; idx=np.arange(len(y))
    for seed in range(N_SEEDS):
        tr,_=next(GroupShuffleSplit(1,test_size=TEST_SIZE,random_state=seed).split(idx,y,gr))
        if len(np.unique(y[tr]))<2: continue
        clf=LogisticRegression(max_iter=MAX_ITER,C=PROBE_C,random_state=seed).fit(X[bl][tr],y[tr])
        p=clf.predict_proba(tf[bl])[:,1]; aucs.append(roc_auc_score(ty,p)); last=p
    dm=fit_diffmean(X[bl],y); pdm=dm.predict_proba(tf[bl])[:,1]
    return float(np.mean(aucs)),float(np.std(aucs)),_bootstrap_auc(np.asarray(ty),np.asarray(last)),float(roc_auc_score(ty,pdm)),last

PROBES=['P_S','P_C','POOLED','I_S','I_C']
res={'note':'resp position; de-inv=max(auc,1-auc); IT resp approx (multi-turn)'}
print('\n=== ROLLOUT transfer (resp): LR mean±std [CI] | de-inv | DiffMean | balAcc@1% ===')
for key in feats:
    ty=labels[key]; tf=feats[key]
    print(f'\n--- {key} (n={len(ty)}, {int((ty==0).sum())}H/{int((ty==1).sum())}D) ---')
    res[key]={}
    for t in PROBES:
        if BL[t] not in tf: continue
        m,s,(lo,hi),dm,p=score(t,tf,ty)
        deinv=max(m,1-m); ba=ood.balanced_accuracy_at_fpr(ty,p,target_fpr=0.01)
        res[key][t]={'lr_auc':m,'lr_std':s,'ci':[lo,hi],'auc_deinv':deinv,'inverted':bool(m<0.5),
                     'diffmean':dm,'balacc_1pct':float(ba)}
        print(f'  {t:>6}: LR={m:.3f}±{s:.3f} [{lo:.3f},{hi:.3f}] de-inv={deinv:.3f}{"(INV)" if m<0.5 else ""}  DM={dm:.3f}  bA@1%={ba:.3f}')

json.dump(res, open(op/'pooled_ood_rollouts.json','w'), indent=2)
print(f'\nSaved {op/"pooled_ood_rollouts.json"}')
