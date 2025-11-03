# Retraining OpenLID with all the new data and changes

`cd baselines/scripts/retrain_openlid`

## OpenLID pipeline

Following [OpenLID's instructions](https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/how_to_update.md) (be cautious, they were not fully up-to-date),  the pipeline is as follows:

That folder contains mostly OpenLID author's scripts with minor changes.

For the baseline family-specific language models, I recommend to run from step 4, skipping cleaning (so that their sampling is not influenced by some minor languages and we don't loose the data). The current cleaning is language-independent, so you can just take `/scratch/project_465002259/eurolid/02-11/openlid_stage2_prep.fasttext` and take only the languages of interest from there.

1. Find additional data and format by the scheme <text>\t<language>\t<source>. If it is an addition to an existing language, it can be appended to it either from a *.parquet or *.tsv using the script `append_to_openlid_parquet.py`.
If the data are for a new language, just convert to a parquet.

2. Data for all languages must be in the same directory.

3. The most recent data (added for some languages, ara_Arab and fas_Arab merged, lat_Latn added, srp_Latn added, zxx_Zxxx added) are at `/scratch/project_465002259/eurolid/02-11-data/`.

4. Cleaning, deduplication, up/downsampling, writing to FastText format and shuffling are done by `make_training_openlid.py`. I was able to run that script on my laptop with only 16 GB of memory, except shuffling. If you fail on memory when shuffling, run `shuf.sh` on LUMI (don't forget to change the hardcoded paths there. They should be changed to variables...) 

When running from scratch, the command is

```commandline
python3 make_training_openlid.py <output_dir> --data_dir <data_dir>
```

If the output of stage 2 from `make_training_openlid.py`, named openlid_stage2_prep.fasttext,  is in <data_dir> directory  and contains only languages of interest, 
the command to run preprocessing will be:

```commandline
python3 make_training_openlid.py <output_dir> --skip_clean --skip_sort --data_dir <data_dir>
```

5. The training on LUMI is run by `lid.sh`. Also beware of the hardcoded paths in the python script! (I hope to fix it today). The hyperparameters are the same as in OpenLID.

## Adding GlotLID data

The data on LUMI are in `/scratch/project_465001890/eurolid/glotlid-corpus/`.

It is also possible download them using the script `download_glotlid.py`.

`make_list_of_glotlid_sources.py` creates the list of GlotLID sources for each language and shows number of samples in GlotLID data.
There is no need to run it, since the resulting list is in `other.tsv` in the root of this repository.

The script `add_from_glotlid.py` shows how to select only the data sources that are of reliable quality and not proprietary. (Beware of hardcoded paths...)
The list of filters there is also for the languages we worked with before;
for Scandinavian etc., if there are some other sources, check their quality and license according to [GlotLID list](https://github.com/cisnlp/GlotLID/blob/main/sources.md).
We also collected licenses of the sources we used [here](https://docs.google.com/spreadsheets/d/162EzUGXDllmujoNG5s_XngSlL4awOJ9F79t5k2OM_FQ/edit?gid=737547198#gid=737547198) at LangID sources sheet.

That script also ensures that wikipedia GlotLID data do not intersect with OpenLID wikipedia data.