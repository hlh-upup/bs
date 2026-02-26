import pickle, json, os, sys
from pathlib import Path
p=Path('experiments/optimized_run1/training_history.pkl')
if not p.exists():
    print('NOT_FOUND')
    sys.exit()
with open(p,'rb') as f:
    hist=pickle.load(f)
# summarize
out={}
for k in ['train_loss','val_loss']:
    arr=hist.get(k, [])
    if arr:
        best=min(arr)
        best_epoch=arr.index(best)+1
        out[k]={'first':arr[0],'last':arr[-1],'best':best,'best_epoch':best_epoch,'len':len(arr)}
val_metrics_list=hist.get('val_metrics',[])
if val_metrics_list:
    best_epoch=out['val_loss']['best_epoch']
    best_metrics=val_metrics_list[best_epoch-1]
    final_metrics=val_metrics_list[-1]
    out['best_metrics']=best_metrics
    out['final_metrics']=final_metrics
print(json.dumps(out, ensure_ascii=False, indent=2))