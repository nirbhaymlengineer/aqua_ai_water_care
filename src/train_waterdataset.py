
"""
cd /Users/nirbhay.borikar/Documents/WaterTest
source .venv_2/bin/activate
"""

# Step 1: Identify and think in advance all necessary libraries

from sklearn.impute import KNNImputer
from torch.utils.data import Dataset # utlities, Dataset 
from torch.utils.data import DataLoader
import torch.nn as nn # neurnal network
import torch.nn.functional as F # functions
import pandas as pd
import torch

import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from torchmetrics import Accuracy 

import joblib

# Step2 : Create a Dataset class to read data, length of it, extract feature from data and store in numpy array, and convert to tensor
class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__() # initialization super constructor method
        df = pd.read_csv(csv_path)
        """
        Before filling missing values, create indicators:

        ph_missing
        Hardness_missing
        Solids_missing
        Chloramines_missing
        Sulfate_missing
        Conductivity_missing
        Organic_carbon_missing
        Trihalomethanes_missing
        Turbidity_missing

        | pH  | ph_missing |
        | --- | ----------- |
        | 7.2 | 0           |
        | NaN | 1           |

        This tells the model: "This value was originally missing."
        """
        # -----------------------
        # Missing Flags
        # -----------------------

        for col in df.columns[:-1]:
            # isna -> NAN
            df[f"{col}_missing"] = (df[col].isna().astype(int))


        """
        KNN Imputation:
        - Instead of filling mean value we use KNN imputation.
        - We estimate missing values using similar records.
        """

        # -----------------------
        # KNN Imputation (K-Nearest Neighbour)
        # -----------------------

        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
        print(df.columns)

        # change labels to numpy as it is not missing 
        self.labels = df_imputed["Potability"].to_numpy()

        self.features = df_imputed.drop(columns=["Potability"]).to_numpy() # convert all column to numpy expect Potability

        # Scaling wit self convert things in to dataset objects, and objects can be used outside class
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        """
        when user enter value you must scales this 
        features in same way as we are doing in training
        ex:
        pH mean = 7.08
        Hardness mean = 196
        Hardness std = 32
        ...
        """


        # Number of features (excluding label)
        self.num_features = self.features.shape[1]

    def __len__(self):
        """ len: result the size of dataset"""
        return len(self.labels) # calculate number of row 

    # now extract features from data
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)# take one argument return features, All columns 
        label = torch.tensor(self.labels[idx], dtype=torch.float32)# only last as it contain label , also convert it to tensor as it loaded in numpy
        
        # self.data[1, -1] # give 1 1 (the last column first element is label )
        #[feature1, feature2, feature3, label]
        return features, label

dataset_train = WaterDataset("water_potability.csv") # class define, and dataset file upload
print(dataset_train.num_features) # print number of column feature variable + missisng varaibles

# Step3: Create a training dataset using dataset.
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
# batch_size is 32, means send 32 random data row to the neural network at one time.
# With this, Uses less memory, Trains faster, and Often generalizes better


# Test it
"""
for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    break
"""

# Step4: Define Neural Network, linear layer for model, and also pass input to layer, and update input layer by layer
class WaterNet(nn.Module):

    def __init__(self, input_size, hidden_layers=[32, 16, 1]):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0]) # define liner layer for model (column(9 + 9 missing), 32 learned features)
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1]) # define liner layer for model (32 patterns -> 16 strongest features)
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2]) # define liner layer for model (16 learned signals -> 1 prediction)

        """
            Input Layer
            (18 features)-> 9 parameter + 9 param missing
                ↓
            Hidden Layer 1
            (32 neurons)
                ↓
            Hidden Layer 2
            (16 neurons)
                ↓
            Output Layer
            (1 neuron)
                ↓
            Potability
            0 or 1

        """


    def forward(self, x):
        """ It describes input parameter, and update it layer by layer"""
        x = torch.relu(self.fc1(x)) # passing to model
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = WaterNet(input_size=dataset_train.num_features,hidden_layers=[32, 16, 1]) # initialize the class.

# Step 5: Loss function

"""
Since output is 0 and 1 binary value, we will use **Binary Cross Entropy** 
- to improve prediction = predict -label
"""
criterion = nn.BCELoss() # define criterion

# Step 6: Optimizer, update input parameters, direction of update depends on gradient sign.
# Adaptive Moment Estimation (Adam): update param + gradient weights + prevent fast decreasing of lr 
optimizer = optim.Adam(model.parameters(),lr=0.001) # 0.001 =nx


# Step 7: Training Loop

num_epochs = 20
"""
- num_epochs it means the model will go through all your training data e.g. 20 times. 
- Each pass gives the model another chance to adjust its weights and reduce the loss. 
"""
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        labels = labels.view(-1,1) 
        """ Reshapes the labels from a flat 1-D tensor of shape (N,) into a 2-D column vector of shape (N, 1).
         -1 means "figure out this dimension automatically" 1 means "one column"""
        optimizer.zero_grad() # we clear the gradients to start from zero for the new batch 
        outputs = model(features) # forward pass: get model's ouputs
        #print(f"model outputs: {outputs[:5]}")
        #print(f"model labels: {labels[:5]}")

        loss = criterion(outputs, labels) # compare the model output to the ground turth labels to compute the loss.       
        #print(f"Loss: {loss.item()}") # DEBUG
        loss.backward() # compute gradients of loss function, this gradients contain direction
        optimizer.step() # pass graindent to optimizer which optimizes it 
        total_loss += loss.item()
        """
        To add up the loss from each batch (e.g. batch 1 loss 0.65, batch 2 loss 0.55 ) during training.
        print(loss)
        tensor(0.5432, grad_fn=...)
        print(loss.item())
        # 0.5432
        """
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss = {total_loss:.4f}")
avg_loss = total_loss / len(train_loader) # it should be at end 
print(f"Average_Loss = {avg_loss:.4f}")


# Step 8: Evaluate Accuracy:

# First, setup the "binary accuracy metric" from torchmetrics.
acc= Accuracy(task="binary")

model.eval() # put the model in evaluation mode with .eval()

with torch.no_grad(): #with no gradient
    for features, labels in train_loader: # iterate over dataloader_test
        outputs =  model(features) # accesing output from forward pass: predicted probabilities
        # now we transform the "predicted probabilities" into "predicted labels"
        preds = (outputs >= 0.5).float() # on the threshold  of 0.5
        # Now update accuracy metric/score.
        acc(preds, labels.view(-1,1))


# Lastly compute the overall accuracy.
accuracy = acc.compute()
# Now print that accuracy
print(f"Accuracy: {accuracy:.2%}")



# Step 9: Save trained model

torch.save(model.state_dict(),"water_model.pth")
print("Model saved successfully.")

"""
The .pth contain "Learned neural-network weights".

example:
Layer1 weights
Layer2 weights
Layer3 weights
Bias values

Everything the model learned during training.
"""

# Step 10: save scaler.(Call scaler using class object, as scaler become dataset object)
# with self  a local variable become attribute (property) of the object.here dataset_train
joblib.dump(dataset_train.scaler,"scaler.pkl")
print("Scaler saved successfully.")

""" 
why save?
Suppose training data was transformed:

Hardness = 200
↓ StandardScaler
0.54

The model learned: Input = 0.54 ,not 200

If later a user enters: Hardness = 200, and we dont scale it 
the input =200, and then prediction become meaningless

"""