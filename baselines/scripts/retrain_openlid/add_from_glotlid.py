import os
from glob import glob
import pandas as pd
from tqdm import tqdm

glotlid_corpus_path = "../../glotlid-corpus/v3.1"
for lang in tqdm((
    'ast_Latn',
    'ban_latn',
    'ben_Beng',
    'bam_Latn',
    'bod_Tibt',
    'cat_Latn',
    'cjk_Latn',
    'dan_Latn',
    'dzo_Tibt',
    'fuv_Latn',
    'heb_Hebr',
    'ilo_latn',
    'kan_Knda',
    'kin_Latn',
    'ktu_Latn',
    'run_Latn',
    'rus_Cyrl',
    'som_Latn',
    'ssw_Latn',
    'sun_Latn',
    'tam_Taml',
    'gug_Latn',
    'lij_Latn'
    )
):
    data = []
    for lang_path in glob(os.path.join(glotlid_corpus_path, lang, "*.txt")):
        if ('_cbc' not in lang_path) and ('_pbc' not in lang_path) and ('_jw' not in lang_path) and (
                '_bloombooks' not in lang_path) \
                and ('ShaShiYaYi' not in lang_path) and ('_Bibles' not in lang_path) and (
                '_bhas' not in lang_path) and ('_wanca' not in lang_path) \
                and ('IN22Conv' not in lang_path) and ('IN22Gen' not in lang_path) and ('JW300' not in lang_path) and (
                'americasnlp2024' not in lang_path) \
                and ('_lti' not in lang_path) and ('mt560' not in lang_path) and ('yuecmnengparallel' not in lang_path) and ("floresdev" not in lang_path) \
                and ('_lyrics' not in lang_path):
            data.append(lang_path)
    if data:
        oldata = pd.read_parquet(f'../data/{lang}.parquet')
        unik = oldata.source.unique()
        print(unik)
        for un in unik:
            seen = False
            for path in data:
                if un in path:
                   data.remove(path)

        print(data)

        for path in data:
            no_data = False
            old_fn = path
            new_fn = os.path.join('../new_data/better_parquets', os.path.basename(path))
            source = os.path.splitext(path.split('_')[-1].rstrip())[0]
            if 'leipzigwiki' in path:
                source = 'leipzigwiki'
                new_fn = os.path.join('../new_data/better_parquets', os.path.basename(path).replace('leipzigwiki', 'leipzigwiki-dedup'))
                old_fn = new_fn
                old_texts_set = set()
                for text in oldata.text:
                    old_texts_set.update(set(text.lower().split('\n')))
                with open(path, 'r') as lw, open(new_fn, 'w') as out:
                    lw_data = set([text.strip() for text in lw.read().lower().split('\n') if text.strip()])
                    lw_data = lw_data - old_texts_set
                    out.write('\n'.join(lw_data))
            if not no_data:
                command = f"cat {old_fn} | awk '" + "{print $0" + f'"\t{lang}\t{source}"' + '}' + f"' > {new_fn.replace('txt', 'tsv')}"
                os.system(command)