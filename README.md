# test-FuSA

```
mamba create -n test-fusa python dvc numpy pandas matplotlib
mamba activate test-fusa
mamba install -c conda-forge dvc-gdrive
dvc init #Inicializando el proyecto dvc
```

```
mamba create -n torch-env python numpy=1.24
mamba activate torch-env
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install ipykernel
```