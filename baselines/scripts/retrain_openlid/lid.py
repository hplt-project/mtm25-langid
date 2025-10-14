import fasttext

model = fasttext.train_supervised('/scratch/project_465002259/OpenLID-v2/openlid_train_sampled_shuffled.fasttext',
                                  minCount=1000, bucket=1000000,
                                  minn=2, maxn=5, lr=0.8, dim=256, epoch=2, thread=68, wordNgrams=1,
                                  )
print(model.predict("lorem ipsum dolor sit amet"))
model.save_model("/scratch/project_465002259/OpenLID-v2/model.bin")