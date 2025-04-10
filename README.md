# Pytorch Implementation of Deep HSMM-Based Text-to-Speech Synthesis
- This software is distributed under the BSD 3-Clause license. Please see LICENSE for more details.
- [Demo samples](https://sp-nitech.github.io/DHSMM-TTS)

# Requirements
- python >= 3.8
- numpy
- pytorch >= 1.11.0 (https://pytorch.org/)
- matplotlib
- tensorboard
- librosa

# Usage

This repository currently provides an implementation of an **acoustic model only**.  
To run training and inference, you must prepare your own **speech database** and **neural vocoder**.  
The included examples assume the use of the **XIMERA Corpus**, which follows this directory structure:  
``/db/ATR-TTS-JP-CORPUS/F009/AOZORAR/T01/000/F009_AOZORAR_00001_T01.wav``


### 1. Data Preparation
Prepare the following files based on your prepared speech database:
- **List files:** `./list/`
- **Phoneme+accent label files:** `./data/phn/phone_hl/`  
👉You can refer to the provided examples in the repository for formatting and structure.


### 2. Generate Mel-Spectrogram Files
```
sh mkmel.sh
```
**output:** ./data/mel/.../filename.npz 

### 3. Training:
**config:** ./scripts/model/demo/config.py
```
sh train.sh
```
**output:** ./model/.../checkpoint_#####

### 4.Inference: 
```
sh gen.sh
```
**output:** ./gen/.../filename.npz (mel-spectrogram files)  
👉The generated mel-spectrograms can be fed into a neural vocoder to synthesize the final waveform.

# Who we are
- Yoshihiko Nankaku (https://www.sp.nitech.ac.jp/~nankaku)
- Takato Fujimoto (https://www.sp.nitech.ac.jp/~taka19)
- Takenori Yoshimura (https://www.sp.nitech.ac.jp/~takenori)
- Shinji Takaki (https://www.sp.nitech.ac.jp/~takaki)
- Kei Hashimoto (https://www.sp.nitech.ac.jp/~hashimoto.kei)
- Keiichiro Oura (https://www.sp.nitech.ac.jp/~uratec)
- Keiichi Tokuda (https://www.sp.nitech.ac.jp/~tokuda)
