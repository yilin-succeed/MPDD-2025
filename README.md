# MPDD-2025

- **environment.yaml**  
  The environment configuration file for setting up the required dependencies.

- **answer_Track2/**  
  Contains the output results of various feature combination experiments as well as the evaluation scores on the testing platform.

- **kfold_checkpoints/**  
  Stores the weight files for different model combinations along with training log files.

- **train.sh**  
  The training script used to start the model training process.

- **test.sh**  
  The testing script used to run model evaluation.

- **train.py**  
  The training code file.

- **test.py**  
  The testing code file.

---

### Feature Combination Definitions

- **3way**  
  Results from combining three types of features.

- **av+feat**  
  Combination of audio-visual features and personality traits.

- **av+user**  
  Combination of audio-visual features and user-level features.

- **av**  
  Audio-visual features only.
