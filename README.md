# Belief State for Visually Grounded, Task-Oriented Neural Dialogue Model

## Preparations
- Clone this repository and change.
```bash
git clone git@github.com:timbmg/believe.git
```
Next, change into the guesswhat directory:
```bash
cd guesswhat
```
- Download the dataset, preprocessed VGG16 FC8 features and vocabulary files.
```bash
bash guesswhat/data/download.sh
```
Alternatively, you can create the VGG16 FC8 yourself with
```bash
cd guesswhat
python3 utils/cache_vgg_features.py
```
For the Vocabulary and Category Vocabulary, if the files are not found in the specified folder for the training scripts, they will be created.

- Download the bin and log files.
```bash
# todo
```

## Baseline Models
First, the baseline models of the Oracle, Guesser and Question Generator have to be trained. The default command line argumnets are those of the orginal paper.
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
