import numpy as np
from typing import List, Dict
from tqdm import tqdm
import torch
from model import Model
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset,DataLoader
import warnings
from transformers import RobertaTokenizerFast
from transformers import AutoModel
from transformers import AdamW


warnings.filterwarnings('ignore')

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()
    return StudentModel(device)


class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self,device):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        self.label_all_tokens = False
        self.device = device
        #print()
        #self.senses_keys = self.get_senses('/home/sameerahmed/Desktop/nlp2023-hw2/model/senses.txt')
        self.senses_keys = self.get_senses('model/senses.txt')
        self.label_to_id = {n: i for i, n in enumerate(self.senses_keys)}
        #print(self.label_to_id)
        self.id_to_label = {i: n for n, i in self.label_to_id.items()}
        self.MAX_LEN = 289

        self.model = self.RobertaModel(len(self.label_to_id.keys()), fine_tune_lm=True)
        self.model.to(self.device)

        #optimizer = AdamW(model.parameters(), lr=5e-5)
        if self.device == 'cpu':
            #self.model.load_state_dict(torch.load('/home/sameerahmed/Desktop/nlp2023-hw2/model/model_checkpoint_epoch-4+1000.pth',map_location=torch.device('cpu')))
            self.model.load_state_dict(torch.load('model/model.pth',map_location=torch.device('cpu')))
        else:
            #self.model.load_state_dict(torch.load('/home/sameerahmed/Desktop/nlp2023-hw2/model/model_checkpoint_epoch-4+1000.pth'))
            self.model.load_state_dict(torch.load('model/model.pth'))
        self.model.eval()
        

    def get_instence_ids(self,data):
        instence_ids_list = []
        for instence_ids in data:
            i_id = []
            for instence_id in instence_ids.keys():
                i_id.append(int(instence_id))
            instence_ids_list.append(i_id)
        return instence_ids_list
    
    def tokens_lower(self,tokens_values):
        tokens = []
        for num in tqdm(range(len(tokens_values))):
            tokens.append(list(map(str.lower,tokens_values[num])))
        return tokens
    
    def get_senses(self,file_path_senses):
        #print(file_path_senses)
        senses = []
        with open(file_path_senses, 'r') as fp:
            for line in fp:         
                x = line[:-1]        
                senses.append(x)
        return senses
    
    class RobertaModel(torch.nn.Module):

        def __init__(self, num_labels: int, fine_tune_lm: bool = True, *args, **kwargs)-> None:

            super().__init__()

            self.num_labels = num_labels
            self.transformer_model = AutoModel.from_pretrained("model/Roberta_Model", output_hidden_states=True)
            if not fine_tune_lm:
                for param in self.transformer_model.parameters():
                    param.requires_grad = False
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(
                self.transformer_model.config.hidden_size, num_labels, bias=False
            )
        def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            labels: torch.Tensor = None,
            compute_predictions: bool = False,
            compute_loss: bool = True,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            # group model inputs and pass to the model
            model_kwargs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
            }
            # not every model supports token_type_ids
            if token_type_ids is not None:
                model_kwargs["token_type_ids"] = token_type_ids
            transformers_outputs = self.transformer_model(**model_kwargs)
            # we would like to use the sum of the last four hidden layers
            transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
            transformers_outputs_sum = self.dropout(transformers_outputs_sum)
            
            logits = self.classifier(transformers_outputs_sum)

            output = {"logits": logits}

            if compute_predictions:
                predictions = logits.argmax(dim=-1)
                output["predictions"] = predictions

            if compute_loss and labels is not None:
                output["loss"] = self.compute_loss(logits, labels)

            return output

        def compute_loss(
            self, logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute the loss of the model.
            Args:
                logits (`torch.Tensor`):
                    The logits of the model.
                labels (`torch.Tensor`):
                    The labels of the model.
            Returns:
                obj:`torch.Tensor`: The loss of the model.
            """
            return F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        
    
        
    class RobertaDataset(Dataset):
        def __init__(self, word_list,label, label_to_id,max_length=289):
            
            self.word_list = word_list
            self.max_length = max_length
            self.label = label
            self.tokenizer = RobertaTokenizerFast.from_pretrained('model/Roberta_Tokenizer',add_prefix_space=True)
            #self.label_all_tokens = False
            self.label_to_id = label_to_id

        def align_label(self,texts, labels,max_length,tokenizer):
        #tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',add_prefix_space=True)
            tokenized_inputs = tokenizer(texts, padding='max_length', max_length=max_length, truncation=True,is_split_into_words=True)
            #print('----------------')
            #print(self.label_to_id)
            word_ids = tokenized_inputs.word_ids()

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(self.label_to_id[labels[word_idx]])
                    except:
                        label_ids.append(-100)
                else:
                    try:
                        label_ids.append(self.label_to_id[labels[word_idx]] if self.label_all_tokens else -100)
                    except:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            return label_ids    

        def __len__(self):
            return len(self.word_list)

        def __getitem__(self, index):
            word = self.word_list[index]
            label = self.align_label(word, self.label[index],self.max_length,self.tokenizer)
            #print(label)

            # Tokenize the word and add special tokens
            encoding = self.tokenizer(
                word,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                is_split_into_words=True
            )

            # Get the input IDs and attention mask
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            #attention_mask = encoding['attention_mask']
            #token_type_ids = encoding['token_type_ids']
            labels = torch.LongTensor(label)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels':labels
            }

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        #print(tokens)
        instance_ids = [sentence['instance_ids'] for sentence in tokens]
        #print(sentences[0]['instance_ids'])
        #print(instance_ids)
        instence_ids_list = self.get_instence_ids(instance_ids)
        #print(instence_ids_list)
        words = [sentence['words'] for sentence in tokens]
        input_words = self.tokens_lower(words)
        #print(len(input_words))
        #print(self.label_to_id)
        test_labels = []
        for instence_ids_t in instence_ids_list:
            test_label = ['__PADDING__']*self.MAX_LEN
            for instence_id_t in instence_ids_t:
                test_label[instence_id_t]='yr.n.h.01'
            test_labels.append(test_label)
        
        test_dataset = self.RobertaDataset(input_words,test_labels,self.label_to_id)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        self.model.eval()
        predictions = []
        #testing_dataloader_check = DataLoader(testing_data)

        with torch.no_grad():
            for i in tqdm(test_dataloader):
                batch = {k: v.to(self.device) for k, v in i.items()}
                #print(batch)
            #print(batch['input_ids'])
                outputs = self.model(**batch, compute_predictions=True)
                y_true = batch["labels"].tolist()
                y_pred = outputs["predictions"].tolist()
                #print(y_true)
                #print(y_pred)

                true_predictions = [
                        [self.id_to_label[p] for (p, l) in zip(pred, gold_label) if l != -100]
                        for pred, gold_label in zip(y_pred, y_true)
                    ]
                predictions += true_predictions
        #print(predictions)
        #print(len(predictions))
        return predictions


