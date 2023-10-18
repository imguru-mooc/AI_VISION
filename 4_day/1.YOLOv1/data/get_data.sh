#!/bin/bash

cd data
							      
# VOC2007 DATASET                                                              
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar 


# Extract tar files
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# Run python file to clean data from xml files
python voc_label.py

# Get train by using train+val from 2007 and 2012
# Then we only test on 2007 test set
# Unclear from paper what they actually just as a dev set
cp 2007_train.txt train.txt
cp 2007_test.txt test.txt

# Move txt files we won't be using to clean up a little bit
mkdir old_txt_files
mv 2007* old_txt_files/

python generate_csv.py

mkdir images
mkdir labels

mv VOCdevkit/VOC2007/JPEGImages/*.jpg images/                                      
mv VOCdevkit/VOC2007/labels/*.txt labels/                                          

# We don't need VOCdevkit folder anymore, can remove
# in order to save some space 
rm -rf VOCdevkit/
mv test.txt old_txt_files/
mv train.txt old_txt_files/
cd ..