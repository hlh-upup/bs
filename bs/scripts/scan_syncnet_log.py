import re,collections
from pathlib import Path
log=Path('prepare_ch_sims_with_features.log')
counts=collections.Counter()
sync_vals=[]
off_vals=[]
with log.open('r',encoding='utf-8',errors='ignore') as f:
  for i,line in enumerate(f):
    low=line.lower()
    if 'syncnet' in low:
      counts['syncnet_lines']+=1
    if 'error' in low: counts['error']+=1
    if 'warn' in low or 'warning' in low: counts['warn']+=1
    m=re.search(r'sync[_ ]?score[:= ]([0-9.]+)',low)
    if m:
      try: sync_vals.append(float(m.group(1)))
      except: pass
    m2=re.search(r'offset[:= ](-?[0-9.]+)',low)
    if m2:
      try: off_vals.append(float(m2.group(1)))
      except: pass
print('统计:',counts)
if sync_vals:
  import numpy as np
  a=np.array(sync_vals)
  print('sync_score count',a.size,'min',a.min(),'max',a.max(),'mean',a.mean(),'std',a.std())
if off_vals:
  import numpy as np
  b=np.array(off_vals)
  print('offset count',b.size,'min',b.min(),'max',b.max(),'mean',b.mean(),'std',b.std())