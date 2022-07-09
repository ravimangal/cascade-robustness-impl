# cascade-robustness
Code and results described in the paper "A Cascade of Checkers for Run-time Certification of Local Robustness"

# Usage
To reproduce the experiments described in the paper run the following commands:

### MNIST
#### For training model:
```
python -u ./code/training.py --experiment=mnist --epsilon=<epsilon> --epochs=100 --batch_size=128 --lr=0.001 --arch=dense_small_3F --gpu=1 --conf_name=<epsilon> > ./experiments/logs/training_mnist_<epsilon>.out
```
#### For certification: 
```
python -u ./code/verify.py --experiment=mnist --conf_name=<epsilon>  --epsilon=<epsilon> --marabou_path=<path_to_Marabou_binary>  --attack=cleverhans > ./experiments/logs/verify_mnist_<epsilon>.out
```

### SafeSCAD
#### For training model:
```
python -u ./code/training.py --experiment=safescad --dataset_file=~/cascade-robustness-impl/experiments/data/safescad/All_Features_ReactionTime.csv --epsilon=<epsilon> --epochs=200 --batch_size=512 --lr=0.01 --arch=safescad --gpu=1 --conf_name=<epsilon> > ./experiments/logs/training_safescad_<epsilon>.out &
```
#### For certification: 
```
python -u ./code/verify.py --experiment=safescad --conf_name=<epsilon>  --epsilon=<epsilon> --marabou_path=<path_to_Marabou_binary>  --attack=cleverhans > ./experiments/logs/verify_safescad_<epsilon>.out 
```

# Experimental Data
All data generated for the experiments is in the `fomlas22_cascade_experimental_data.zip` file available [here](https://drive.google.com/file/d/1VX5zPA9fpwrxi7w4f28ZDPqJLVDG-JEU/view?usp=sharing). This includes the train and test datasets, trained models, queries issued to Marabou, and the output logs. The output logs `verify_mnist_<epsilon>.out` and `verify_safescad_<epsilon>.out` in the `logs` folder contain the final statistics that are reported in the paper. 

# Dependencies
- Gloro code from https://github.com/klasleino/gloro (commit 5ebfe0f3850bca20e4ee4414fa2ee8a4af303023)
- Marabou binary compiled from source available at https://github.com/NeuralNetworkVerification/Marabou
- Tensorflow 2.6.2, pandas, numpy, scriptify, cleverhans, autoattack, scipy

