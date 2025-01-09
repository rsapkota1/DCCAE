import numpy as np
import pandas as pd
import torch
import os
import pytorch_lightning as pl
from cca_zoo import DCCAE
from cca_zoo.deepmodels import architectures
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from numpy_dataset import NumpyDataset
import random
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import pickle


# Set seed for NumPy
seed = 42
np.random.seed(seed)
random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df_gray_PCA=pd.read_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/Preprocess/PCA/GM/gray_transformation_100.csv').iloc[:,1:]
df_white_PCA=pd.read_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/Preprocess/PCA/FA/white_transformation_100.csv').iloc[:,1:]

scaler = StandardScaler()
df_gray_norm = pd.DataFrame(scaler.fit_transform(df_gray_PCA), columns=df_gray_PCA.columns)
df_white_norm = pd.DataFrame(scaler.fit_transform(df_white_PCA), columns=df_white_PCA.columns)


df_gray_PCA_narray=df_gray_norm.values
df_white_PCA_narray=df_white_norm.values

full_data = NumpyDataset([df_gray_PCA_narray,df_white_PCA_narray])
# Save full_data to a file
with open("/data/users4/rsapkota/DCCA_AE/REDO_CODE/DCCA/Data_Loader/full_data.pkl", "wb") as f:
    pickle.dump(full_data, f)
#full_data.to_pickle('/data/users4/rsapkota/DCCA_AE/REDO_CODE/DCCA/Data_Loader/full_data.pkl')

alldata_loader = torch.utils.data.DataLoader(full_data)


train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size])


train=int(0.8 * len(train_dataset))
test = len(train_dataset) - train
train_train_dataset, test_test_dataset = torch.utils.data.random_split(train_dataset, [train, test])

train_loader_train = torch.utils.data.DataLoader(train_train_dataset, batch_size=256)
val_loader_train = torch.utils.data.DataLoader(test_test_dataset, batch_size=256)
test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=256)

LATENT_DIMS=20
encoder_1 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=100, layer_sizes=(100,))
encoder_2 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=100, layer_sizes=(100,))
decoder_1 = architectures.Decoder(latent_dimensions=LATENT_DIMS, feature_size=100, layer_sizes=(100,))
decoder_2 = architectures.Decoder(latent_dimensions=LATENT_DIMS, feature_size=100, layer_sizes=(100,))

#For Ablation Study
# LATENT_DIMS=20
# encoder_1 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=100)
# encoder_2 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=100)
# decoder_1 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=100)
# decoder_2 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=100)


EPOCHS=50
dcca = DCCAE(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2],decoders=[decoder_1, decoder_2],lr=0.01, lam=0.04, latent_dropout=0)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary= True

)

trainer.fit(dcca, train_loader_train, val_dataloaders=[val_loader_train])

train_sc=dcca.score(train_loader_train)
print(train_sc)

validation_sc=dcca.score(val_loader_train)
print(validation_sc)

test_sc=dcca.score(test_loader)
print(test_sc)

