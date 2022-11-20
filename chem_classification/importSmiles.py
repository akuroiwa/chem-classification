"""
Create a dataset in json file format from smiles.
"""

import os
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import BRICS

def readSmiles(target, path, regression=False):
    train_df = pd.DataFrame(columns=["text_a", "text_b", "labels"])
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            # train_df = train_df.append(importSmiles(target, file_path, regression), ignore_index=True)
            df = pd.DataFrame(importSmiles(target, file_path, regression), columns=["text_a", "text_b", "labels"])
            train_df = pd.concat([train_df, df], ignore_index=True)
    train_df.to_json(os.path.join(path, "smiles.json"))

def importSmiles(target, smiles_file, regression=False):
    # smiles = open(smiles_file)
    # train_df = pd.DataFrame(columns=["text", "labels"])

    target_token = set()
    target_token.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(target)))
    target_token_str = ' '.join(target_token)
    # gen_df = pd.read_csv(smiles_file, index_col=0, usecols=[1, 2])
    # gen_df = pd.read_csv(smiles_file, index_col=0, usecols=[1, 2], names=["smiles", "dice_similarity"])
    gen_df = pd.read_csv(smiles_file, index_col=0)
    if regression:
        labels = gen_df["dice_similarity"]
    else:
        labels = []
        for i in gen_df["dice_similarity"]:
            if i <= 0.5:
                l = 0
            elif i > 0.5:
                l = 1
            else:
                l = -1
            labels.append(l)
    smiles = []
    for s in gen_df["smiles"]:
        s_token = set()
        s_token.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(s)))
        s_token_str = ' '.join(s_token)
        smiles.append(s_token_str)
    train_df = pd.DataFrame({"text_a": target_token_str, "text_b": smiles, "labels": labels})
    return train_df

def create_json_for_train_and_eval(target, train_smiles="train-smiles", eval_smiles="eval-smiles", regression=False):
    # Preparing train data
    readSmiles(target, train_smiles, regression)

    # Preparing eval data
    readSmiles(target, eval_smiles, regression)


def console_script():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--target", dest='target', default=None, type=str, help="Target smile.")
    parser.add_argument("-p", "--path", dest='path', default="train-smiles", type=str, help="Directory from which you want to read.  Default is train-smiles.")
    parser.add_argument("-r", "--regression", dest='regression', action='store_true', help="If True, Outputs dice similarity as labels for SimilarityRegression.")
    args = parser.parse_args()

    # readSmiles(args.path)
    readSmiles(args.target, args.path, args.regression)

if __name__ == "__main__":
    console_script()
