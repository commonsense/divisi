from csc.conceptnet4.analogyspace import conceptnet_by_relations, identities_for_all_relations
from csc.divisi.blend import Blend
from csc.divisi import export_svdview

byrel = conceptnet_by_relations('en')
t=identities_for_all_relations(byrel)
b=Blend(byrel.values()+[t])
s=b.svd()
export_svdview.write_packed(s.u, 'littleblend', lambda x:x)
s.summarize()
