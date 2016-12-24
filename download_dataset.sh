cd data
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip captions_train-val2014.zip
rm train2014.zip val2014.zip captions_train-val2014.zip
mv captions_train-val2014/captions_train2014.json annotations/
mv captions_train-val2014/captions_val2014.json annotatios/
