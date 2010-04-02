import codecs, gzip
utf8_reader = codecs.getreader('utf-8')

def gzip_open_utf8(filename):
    return utf8_reader(gzip.open(filename))

rejected_types=(
    'aux',
    'auxpass',
    'cop',
    'attr',
    'xcomp',
    'mark',
    'rel',
    'acomp',
    'expl',
    'tmod',
    'num',
    'number',
    'abbrev',
    'advmod',
    'neg',
    'poss',
    'det',
    'prt',
    'prep',
)

def iter_relations(filename):
    rel_input = gzip_open_utf8(filename)
    for line in rel_input:
        line = line.strip()
        parts = line.split()
        if len(parts) != 3: continue
        rel = parts[1]
        if rel in rejected_types: continue
        yield parts
    rel_input.close()

def filter_rels(filename):
    rel_dict = {}
    for left, rel, right in iter_relations(filename):
        cur_val = rel_dict.get(left,0)
        rel_dict[left] = cur_val+1
        cur_val = rel_dict.get(right,0)
        rel_dict[right] = cur_val+1
    outfile = codecs.open('evenless.rel', 'w', encoding="utf-8")
    for left, rel, right in iter_relations(filename):
        if rel_dict[left] < 15: continue
        if rel_dict[right] < 15: continue
        outfile.write(' '.join((left, rel, right)) + '\n')
    outfile.close()
    return rel_dict

def count_relations(filename):
    rel_dict = {}
    for left, rel, right in iter_relations(filename):
        cur_val = rel_dict.get(rel,0)
        rel_dict[rel] = cur_val+1
    return rel_dict

if __name__ == "__main__":
    print filter_rels('all.rel.gz')
