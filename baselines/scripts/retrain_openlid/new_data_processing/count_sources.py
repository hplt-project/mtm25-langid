import argparse
import os
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--first', default='lij')
parser.add_argument('--second', default='oci')

args = parser.parse_args()
# pq = os.path.expanduser(f"~/PycharmProjects/OpenLID-v2/new_data/oci_only/{args.first}_Latn.parquet")
#
# pq2 = os.path.expanduser(f"~/PycharmProjects/OpenLID-v2/new_data/oci_only/{args.second}_Latn.parquet")
#
# new_data = pd.read_parquet(pq)
# new_data2 = pd.read_parquet(pq2)
# #for source in new_data.source.unique():
# source = 'lti'
# print(source)
# out = os.path.expanduser(f"~/PycharmProjects/OpenLID-v2/new_data/{args.first}_Latn_no_{source}-pilar")
out = os.path.expanduser('~/PycharmProjects/OpenLID-v2/new_data/oci_no_pilar_frp')
# shutil.rmtree(out, ignore_errors=Tru
# os.makedirs(out, exist_ok=True)
# new_data_df = new_data[new_data.source != source].reset_index(drop=True)
# new_data_df.to_parquet(out+f"/{args.first}_Latn.parquet", compression="gzip")
# source = "pilar"
# new_data_df = new_data2[new_data2.source != source].reset_index(drop=True)
# new_data_df.to_parquet(out + f"/{args.second}_Latn.parquet", compression="gzip")
out_train = os.path.join(out, 'train')
# shutil.rmtree(out_train, ignore_errors=True)
# os.makedirs(out_train, exist_ok=True)

#os.system(f"python3 scripts/retrain_openlid/make_training_openlid.py {out_train} --data_dir {out}")
#os.system(f"python3 scripts/retrain_openlid/lid.py --data_dir {out_train}")
os.system(f"python3 scripts/fasttext_predictions.py --dataset udhr --model retrained --languages-file data/OpenLID-v2/languages.txt --enable-preprocessing --model-path {out_train}/model.bin --out_path {out_train}/results.jsonl")
os.system(f"python3 scripts/evaluate.py {out_train}/results.jsonl --model retrained --languages-file data/OpenLID-v2/languages.txt --dataset udhr > {out_train}/eval_udhr.jsonl")