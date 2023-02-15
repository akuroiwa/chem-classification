from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import torch

from rdkit import Chem
from rdkit.Chem import BRICS
import os

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

class SimilarityClassification(object):

    def __init__(self, output_dir='chem-outputs/'):

        # Optional model configuration
        self.model_args = ClassificationArgs(num_train_epochs=1)
        # self.model_args.reprocess_input_data = True
        self.model_args.output_dir = output_dir
        self.model_args.overwrite_output_dir = True
        # self.model_args.use_cached_eval_features = True
        # self.model_args.silent = True
        # self.model_args.logging_steps = 0

        self.cuda_available = torch.cuda.is_available()

        # Create a ClassificationModel
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            self.model = ClassificationModel(
                # "roberta",
                'electra',
                output_dir,
                num_labels=3,
                args=self.model_args,
                use_cuda=self.cuda_available
            )
        else:
            self.model = ClassificationModel(
                # "roberta",
                # "roberta-base",
                'electra',
                'google/electra-small-discriminator',
                # 'google/electra-base-discriminator',
            # self.model = ClassificationModel(
            #     'bert',
            #     'bert-base-cased',
                num_labels=3,
                args=self.model_args,
                use_cuda=self.cuda_available
            )

    def train_and_eval(self, train_json, eval_json):
        train_df = pd.read_json(train_json)
        # Train the model
        self.model.train_model(train_df)
        # self.model.train_model(train_df, verbose=False)

        eval_df = pd.read_json(eval_json)
        # Evaluate the model
        # result, model_outputs, wrong_predictions = self.model.eval_model(eval_df)
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_df, verbose=False, silent=True)

    def predict_smiles_pair(self, text_a, *text_b):
        if text_b:
            target = text_a

            text_b_list = text_b[0].split(' ')
            if len(text_b) > 1:
                text_b_list += list(text_b[1:])

        elif type(text_a) is list:
        # else:
            target = text_a[0]
            text_b_list = text_a[1:]

        target_set = set()
        target_set.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(target)))
        target_token = ' '.join(target_set)

        smiles_set = set()
        for smiles in text_b_list:
            smiles_set.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles)))
        smiles_token = ' '.join(smiles_set)

        # Make predictions with the model
        prediction, raw_outputs = self.model.predict([[target_token, smiles_token]])
        return prediction, raw_outputs


class SimilarityRegression(SimilarityClassification):

    def __init__(self, output_dir='chem-reg-outputs/'):

        # Optional model configuration
        self.model_args = ClassificationArgs(num_train_epochs=1)
        # self.model_args.reprocess_input_data = True
        self.model_args.output_dir = output_dir
        self.model_args.overwrite_output_dir = True
        # self.model_args.use_cached_eval_features = True
        # self.model_args.silent = True
        # self.model_args.logging_steps = 0
        self.model_args.regression = True

        self.cuda_available = torch.cuda.is_available()

        # Create a ClassificationModel
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            self.model = ClassificationModel(
                # "roberta",
                'electra',
                output_dir,
                num_labels=1,
                args=self.model_args,
                use_cuda=self.cuda_available
                )
        else:
            self.model = ClassificationModel(
                # "roberta",
                # "roberta-base",
                'electra',
                'google/electra-small-discriminator',
            # self.model = ClassificationModel(
            #     'bert',
            #     'bert-base-cased',
                num_labels=1,
                args=self.model_args,
                use_cuda=self.cuda_available
                )
