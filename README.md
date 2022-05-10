# cascade-robustness
Code and results described in the paper "A Cascade of Checkers for Run-time Certification of Local Robustness"

# Usage
To reproduce the experiments described in the paper run the following commands:

### MNIST
#### For training model:
```
python -u ./code/training.py --experiment=mnist --epsilon=0.3 --epochs=100 --batch_size=128 --lr=0.001 --arch=dense_small_3F --gpu=1 --conf_name=small > ./experiments/logs/training_mnist_small.out
```
#### For certification: 
```
python -u ./code/verify.py --experiment=mnist --conf_name=small  --epsilon=0.3 --marabou_path=<path_to_Marabou_binary>  --attack=cleverhans > ./experiments/logs/verify_mnist_small_bin.out
```

### SafeSCAD
#### For training model:
```
python -u ./code/training.py --experiment=safescad --dataset_file=~/cascade-robustness-impl/experiments/data/safescad/All_Features_ReactionTime.csv --epsilon=0.05 --epochs=200 --batch_size=512 --lr=0.01 --arch=safescad --gpu=1 --conf_name=base_bin > ./experiments/logs/training_safescad_base.out &
```
#### For certification: 
```
python -u ./code/verify.py --experiment=safescad --conf_name=base_bin  --epsilon=0.05 --marabou_path=/home/ravi/Marabou-orig/build/bin/Marabou  --attack=cleverhans > ./experiments/logs/verify_safescad_base_bin.out 
```

# Experimental Data
All data generated for the experiments is in the `fomlas22_experimental_data.zip` file available [here](https://drive.google.com/file/d/1msw-D6gTcIS2d0z-Xg6AuAIeyenlk9B-/view?usp=sharing). This includes the train and test datasets, trained models, queries issued to Marabou, and the output logs. The output logs `verify_mnist_small_bin.out` and `verify_safescad_base_bin.out` contain the final statistics that are reported in the paper. 

# Dependencies
- Gloro code from https://github.com/klasleino/gloro (commit 5ebfe0f3850bca20e4ee4414fa2ee8a4af303023)
- Marabou binary compiled from source available at https://github.com/NeuralNetworkVerification/Marabou
- Tensorflow 2.6.2, pandas, numpy, scriptify, cleverhans, autoattack, scipy

