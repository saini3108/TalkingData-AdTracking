#!/bin/bash

DIRECTORY="download"

if [ ! -d "$DIRECTORY"]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "download dir will be created"
  mkdir download
  echo "download dir was created"
fi
echo "There is already a download dir"
echo "What will be deleted and recreated"
rm -rf ./download
mkdir download
echo "download dir was created"

DIRECTORY="data"
if [ ! -d "$DIRECTORY"]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "data dir will be created"
  mkdir data
  echo "data dir was created"
fi
echo "There is already a data dir"
echo "What will be deleted and recreated"
rm -rf ./data
mkdir data
echo "data dir was created"

DIRECTORY="input"
if [ ! -d "$DIRECTORY"]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "input dir will be created"
  mkdir input
  echo "input dir was created"
fi
echo "There is already a input dir"
echo "What will be deleted and recreated"
rm -rf ./input
mkdir input

echo "data dir was created"
echo "Downloading data is starting."

wget "https://www.kaggle.com/c/8540/download/train_sample.csv.zip" -P ./download
wget "https://www.kaggle.com/c/8540/download/sample_submission.csv.zip" -P ./download
wget "https://www.kaggle.com/c/8540/download/test.csv.zip" -P ./download
wget "https://www.kaggle.com/c/8540/download/train.csv.zip" -P ./download

echo "Download finished"
echo "Unzipping files and cleanup"
unzip ./download/sample_submission.csv.zip -d data/
unzip ./download/train_sample.csv.zip -d data/
unzip ./download/train.csv.zip -d data/
unzip ./download/test.csv.zip -d data/

cp -a ./data/mnt/ssd/kaggle-talkingdata2/competition_files/. ./input/
cp -a ./data/*.csv ./input/
rm -rf ./data
echo "temp data dir was deleted"