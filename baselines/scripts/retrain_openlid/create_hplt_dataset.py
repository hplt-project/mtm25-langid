import json
import os
from glob import glob

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

out = '../../data/hplt'
os.makedirs(out, exist_ok=True)
scripts = {}
ids = {'sq': 'als'}
with open("../../data/OpenLID-v2/languages.txt", 'r') as list_f:
    for line in list_f:
        spl = line.split('_')
        scripts[spl[0]] = spl[1].rstrip()
languages = set()
with open(os.path.join(out, 'test.jsonl'), 'w') as hplt_f:
    for lang in glob("../../../../benchmarks/langid/testsets/gold/*"):
        try:
            code = Lang(lang.split(os.extsep)[-1])
        except InvalidLanguageValue as e:
            print(e)
            continue
        with open(lang, 'r') as lang_f:
            for line in lang_f:
                try:
                    script = scripts[code.pt3]
                    iso_639 = code.pt3
                except KeyError:
                    iso_639 = ids[code.pt1]
                    script = scripts[iso_639]
                hplt_f.write(
                    json.dumps(
                        {
                            "id": iso_639 + "_" + script,
                            "sentence": line.strip(),
                            "language": code.name,
                            "iso639-3": iso_639,
                            "iso15924": script,
                        },
                    ) + '\n',
                )
                languages.add(iso_639 + "_" + script)
languages = list(languages)
languages.sort()
with open('../../language-lists/hplt.txt', 'w') as language_list_out:
    for lang in languages:
        language_list_out.write(lang + "\n")
