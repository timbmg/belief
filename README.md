# Belief State for Visually Grounded, Task-Oriented Neural Dialogue Model

## Preparations
### Requirements
This code as been developed with python 3.6.5.
Requirements can be found in the [requirements.txt](requirements.txt).

### Downloads
- Clone this repository.
```bash
git clone git@github.com:timbmg/believe.git
```
Next, change into the guesswhat directory:
```bash
cd guesswhat
```
- Download the dataset, preprocessed VGG16 FC8 features and vocabulary files.
```bash
bash data/download.sh
```
Alternatively, you can create the VGG16 FC8 yourself with
```bash
python3 utils/cache_vgg_features.py
```
Note that you need the training and validation set of MS COCO 2014.  
For the Vocabulary and Category Vocabulary, if the files are not found in the specified folder for the training scripts, they will be created.

- Download the bin and log files of the pretrained models.
```bash
# todo
```

## Baseline Models
First, the baseline models of the Oracle, Guesser and Question Generator have to be trained. The default command line arguments are those of the original paper.
- Oracle
```bash
python3 trainOracle.py
```
- Guesser
```bash
python3 trainGuesser.py
```
- Question Generator
```bash
python3 trainQGen.py
```

## Belief Models
### Training
The different belief models can be trained with the following commands. Note that these are the settings that achieved the best results on the validation set. For all other hyperparameters we refer to the help text in the file.
- Belief
```bash
python3 trainQGenBelief.py \
  --object-embedding-setting learn-emb-spatial \
  --visual-embedding-dim 0 \
  --category-embedding-dim 256
```
- Belief+FineTune
```bash
python3 trainQGenBelief.py \
  --train-guesser-setting \
  --object-embedding-setting learn-emb-spatial \
  --visual-embedding-dim 128 \
  --category-embedding-dim 512
```
- Visual Attention  
First the ResNet features have to be cached. This can be done with the following script (warning: the file will be about 69GB large).
```bash
python3 cache_resnetblock3_features.py
```
```bash
python3 trainQGenBelief.py \
  --train-guesser-setting \
  --visual-representation resnet-mlb \
  --object-embedding-setting learn-emb-spatial \
  --visual-embedding-dim 128 \
  --category-embedding-dim 512
```

### Evaluation
All Question Generator models can be evaluated with the inference script.
If the model is a belief model, the `-belief` option has to be passed addionally. In case the model shall be evaluated on the test set, please pass the `-split test` option. Further, in order to save the outputs (i.e. generated dialogues, belief and guesser probabilities), please pass `-save`. For all other options we refer to the script.
```bash
python3 inference.py -bin/$qgenfile.pt
```
## Analysis
### Stats and Graphs
All analysis can be reproduced from the [analysis jupyter notebook](guesswhat/analysis/analysis.ipynb). The log files are created by passing the `-save` option to the inference script or can be downloaded as mentioned in the Preparations section.
### Web Tool
In order to run the web tool for comparing dialogues download the http-server from npm [here](https://www.npmjs.com/package/http-server).
If the log files are not in the analysis directory, move them there. Then change to analysis directory and start the http-server.
```bash
cd analysis
http-server
```
