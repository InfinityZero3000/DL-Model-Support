import re

BASE = '/Users/nguyenhuuthang/Documents/RepoGitHub/LexiLingo/DL-Model-Support/datasets/wi+locness/m2/'

files = {
    'A.train': BASE + 'A.train.gold.bea19.m2',
    'A.dev':   BASE + 'A.dev.gold.bea19.m2',
    'B.train': BASE + 'B.train.gold.bea19.m2',
    'B.dev':   BASE + 'B.dev.gold.bea19.m2',
    'C.train': BASE + 'C.train.gold.bea19.m2',
    'C.dev':   BASE + 'C.dev.gold.bea19.m2',
    'N.dev':   BASE + 'N.dev.gold.bea19.m2',
}

error_types = {}
for name, path in files.items():
    with open(path) as f:
        content = f.read()
    sentences = [b for b in content.strip().split('\n\n') if b.strip()]
    annotations = re.findall(r'A \d+ \d+\|\|\|([\w:]+)\|\|\|', content)
    for a in annotations:
        error_types[a] = error_types.get(a, 0) + 1
    print(f'{name}: {len(sentences)} sentences, {len(annotations)} annotations')

print()
print('Top 15 error types:')
for k, v in sorted(error_types.items(), key=lambda x: -x[1])[:15]:
    print(f'  {k}: {v}')
