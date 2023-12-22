import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler 

# ignore warnings regarding "not removing hydrogen atom without neighbors" 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def preprocessing(df):
    
    smiles_list = df["smiles"].tolist()
    
    ecfps= []
    mols = []

    for smiles in smiles_list:
        
        ### create mol objects
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

        ### create ECFP fingerprints -> shape (1427,2048)
        fp_sparseVec = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
    
        fp = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fp_sparseVec, fp)
        
        ecfps.append(fp)
    
    # check number of mols and shape of fingerprints
    print(f" nMols: {len(mols)}")
    
    ### create descriptors -> shape (1427, 200 )
    
    filter = list(range(17,25))
    real_descr = [i for i in range(208) if i not in filter]
    total_mols = len(mols)
    
    rdkit_descriptors = []
    
    for i,mol in enumerate(mols):

        if i % 400 == 0:
            print(f'... processing mol {i} of {total_mols}')
    
        descrs = list()
        for descr in Descriptors._descList:
            _, descr_calc_fn = descr
            descrs.append(descr_calc_fn(mol)) 
        
        descrs = np.array(descrs) # creates vec of shape 208
        descrs = descrs[real_descr] # uses only 200 "important" descrs # ask why 200
        rdkit_descriptors.append(descrs) # creates a nested list of descr-vecs

    rdkit_descriptors = np.array(rdkit_descriptors) # convert to numpy 

    print("... done") 
    
    #return rdkit_descriptors

#def scale(rdkit_descriptors): ### to be continued
    
    
    
    ### compute quantils and scale desriptors # only train data!
    
    rdkit_descriptors_quantils = np.zeros_like(rdkit_descriptors)
    
    for column in range(rdkit_descriptors.shape[1]):
        raw_values_ecdf = rdkit_descriptors[:,column].reshape(-1) # train
        raw_values = rdkit_descriptors[:,column] # val,test Or train

        ecdf = ECDF(raw_values_ecdf)
        quantils = ecdf(raw_values)
        rdkit_descriptors_quantils[:,column] = quantils
        
    print(f" min and max quantil: {np.min(rdkit_descriptors_quantils),np.max(rdkit_descriptors_quantils)}")
    
    ### Stack and Scale -> shape (1427,2248)
    scaler = StandardScaler()
    
    data = np.hstack([ecfps, rdkit_descriptors_quantils])
    data = scaler.fit_transform(data)
    
    print(f" min and max of data after scaling: {np.min(data),np.max(data)}")
    print(f"data.shape: {data.shape}")
    
    return data
    
        
        
        