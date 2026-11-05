"""Register-buffer ownership for cross-low/high prefetch, not a GPU proof."""
checks=0
for mi in [1,2,4,8,16]:
 for batch in [1,2,4,8,16]:
  if batch>mi or mi%batch:continue
  groups=mi//batch
  for stages in range(66):
   barriers=0
   for stage in range(stages):
    buffers=[(0,0),None];consumed=set();reads=[(0,0)]
    for half in range(2):
     for g in range(groups):
      slot=(half*groups+g)%2
      assert buffers[slot]==(half,g)
      nxt=(half,g+1) if g+1<groups else (1,0) if half==0 else None
      if nxt:
       prior=buffers[slot^1]
       assert prior is None or prior in consumed
       buffers[slot^1]=nxt;reads.append(nxt)
      assert buffers[slot]==(half,g)
      consumed.add((half,g));checks+=1
    assert reads==[(h,g) for h in range(2) for g in range(groups)]
    barriers+=1
   assert barriers==stages
print('CROSS_HALF_REGISTER_PROTOCOL_PASS',checks)
