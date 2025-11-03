Retraining OpenLID with all new data and changes:

Following [OpenLID's instructions](https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/how_to_update.md) (be cautious, they were not fully up-to-date),  the pipeline is as follows:

`cd baselines/scripts/retrain_openlid`

That folder contains mostly OpenLID author's scripts with minor changes.

For the baseline family-specific language models, I recommend to run from step 4, skipping cleaning (so that their sampling is not influenced by some minor languages and we don't loose the data). The current cleaning is language-independent, so you can just take `` and take only the languages of interest from there.

1. Find additional data and format by the scheme <text>\t<language>\t<source>. If it is an addition to an existing language, it can be appended to it either from a *.parquet or *.tsv using the script `append_to_openlid_parquet.py`.
If the data are for a new language, just convert to a parquet.

2. Data for all languages must be in the same directory.

3. The most recent data (added for some languages, ara_Arab and fas_Arab merged, lat_Latn added, srp_Latn added, zxx_Zxxx added) are at `/scratch/project_465002259/eurolid/02-11-data/`.

4. Cleaning, deduplication, up/downsampling, writing to FastText format and shuffling are done by `make_training_openlid.py`. I was able to run that script on my laptop with only 16 GB of memory, except shuffling. If you fail on memory when shuffling, run `shuf.sh` on LUMI (don't forget to change the hardcoded paths there. They should be changed to variables...) 

5. The training on LUMI is run by `lid.sh`. Again, beware of the hardcoded paths in the python script! (I hope to fix it today). The hyperparameters are the same as in OpenLID.