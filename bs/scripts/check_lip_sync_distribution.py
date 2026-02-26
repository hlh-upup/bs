import pickle, numpy as np, sys
p = sys.argv[1] if len(sys.argv)>1 else 'datasets/ac_final_processed_lipsync.pkl'
d = pickle.load(open(p,'rb'))
for sp in ['train','val','test']:
    if sp not in d: continue
    lab_new = d[sp]['labels'].get('lip_sync_score_new')
    vm_new  = d[sp]['valid_masks'].get('lip_sync_score_new')
    lab_old = d[sp]['labels'].get('lip_sync_score')
    vm_old  = d[sp]['valid_masks'].get('lip_sync_score')
    def stats(tag, lab, vm):
        if lab is None: print(f'  {tag}: MISSING'); return
        arr_all = np.array(lab, float)
        mask = np.array(vm, bool) if vm is not None else np.ones_like(arr_all, bool)
        arr = arr_all[mask]
        if arr.size == 0:
            print(f'  {tag}: valid=0'); return
        print(f'  {tag}: valid={arr.size} unique={len(np.unique(arr))} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}')
    print(f'== {sp} ==')
    stats('OLD', lab_old, vm_old)
    stats('NEW', lab_new, vm_new)
    if lab_new is not None and lab_old is not None and len(lab_new)==len(lab_old) and len(lab_new)>1:
        a=np.array(lab_new,float); b=np.array(lab_old,float)
        aa=a-a.mean(); bb=b-b.mean()
        denom=(aa@aa)**0.5*(bb@bb)**0.5
        corr=(aa@bb)/denom if denom>0 else 0.0
        print(f'  Pearson(new,old)={corr:.4f}')