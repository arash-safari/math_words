# %%
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup,TrainingArguments, Trainer
from tqdm import tqdm, trange
import numpy as np 
import random 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os 
from torch.nn.functional import softmax
from torch import argmax
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
# %%
# download database gsm8k 

tokenizer = None

# %%
# dataset
def get_formated_body_answer(answer):
    return '\n'.join(answer[:-1].split('\n'))

def get_fomated_final_answer(answer):
    return answer.split('\n')[-1]

def get_formated_question(question, PROMPT_ANSWER_TOKEN="A: "):
    return  question+"?\n"+PROMPT_ANSWER_TOKEN    



def get_formated_finetune(item):
        return get_formated_question(item["question"]) + \
                get_formated_body_answer(item["answer"]) + \
                get_fomated_final_answer(item["answer"]) + \
                tokenizer.eos_token

def get_formated_test(item):
        return get_formated_question(item["question"])
    
def get_batch_formated_finetune(batch):
    return [get_formated_finetune(item) for item in batch]

def get_batch_formated_test(batch):
    return [get_formated_test(item) for item in batch]

def finetune_collate_tokenizer(batch):
    input_text = get_batch_formated_finetune(batch)        
    tokens = tokenizer(input_text, 
                            padding=True,
                            truncation=True,
                            return_tensors='pt')
    return {**tokens, 
            "question":[item["question"] for item in batch],
            "answer":[item["answer"] for item in batch]}
    
def validation_collate_tokenizer(batch):
    input_text = get_batch_formated_test(batch)        
    tokens = tokenizer(input_text, 
                            padding=True,
                            truncation=True,
                            return_tensors='pt')

    return {**tokens, 
            "question":[item["question"] for item in batch],
            "answer":[item["answer"] for item in batch]}

    
class TextGenerationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples["question"])
    
    def __getitem__(self, idx):
        
        # Tokenize and encode the input and target text
        
        item = self.samples.select([idx]).to_dict()
        # item["input_ids"] = input_ids[0]
        # target_ids = tokenizer.encode(target_text, padding='max_length',truncation=True, return_tensors='pt',max_length=1024)
        return {key:item[key][0] for key in item}



# %%
class GPT2Finetuner(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        
    def forward(self, input_ids,attn_masks):
        return  self.gpt2(input_ids, labels=input_ids,attention_mask=attn_masks)
    
    
    def is_generated_answer_correct(self, generated_answer,final_result):
        if not self.tokenizer.eos_token in generated_answer:
            return False
    
        generated_answer = generated_answer.split(self.tokenizer.eos_token)[0]
        if "####" in generated_answer:
            if generated_answer.rsplit("####",1)[1].strip().lower()==final_result.strip().lower():
                return True
        return False
    
    
    def correctness_genrations(self, output, samples):
        final_results = []
        for answer in samples["answer"]:
            final_results.append(answer[-1])
        argmx = argmax(output.logits,dim=-1)
        generated_answers = self.tokenizer.batch_decode(argmx)
        total_correct = 0
        for generated_answer, final_result in zip(generated_answers,final_results):
            if self.is_generated_answer_correct( generated_answer,final_result):
                total_correct +=1
        return total_correct
    
    def training_step(self, batch, batch_idx):

        output = self(batch['input_ids'],batch['attention_mask'])

        train_loss = output.loss

        correct = self.correctness_genrations(output,batch)
        
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
            # info to be used at epoch end 
            "correct": correct,
            "total": len(batch['input_ids'])
        }

        return batch_dictionary

    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
 
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
 
        # creating log dictionary
        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)
         
        self.logger.experiment.add_scalar("Accuracy/Train",
                                            correct/total,
                                            self.current_epoch)
 
        epoch_dictionary={
            # required
            'loss': avg_loss}
        print(epoch_dictionary)
 
        # return epoch_dictionary

    
    def validation_step(self, batch, batch_ids):
        
        print("input_ids_size",batch['input_ids'].size())
        output = self(batch['input_ids'],batch['attention_mask'])
        
        train_loss = output.loss

        correct = self.correctness_genrations(output,batch)
        
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
                
                
            # info to be used at epoch end 
            "correct": correct,
            "total": len(batch)
        }

        return batch_dictionary

    def validation_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
 
        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
 
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
 
        # creating log dictionary
        self.logger.experiment.add_scalar("Loss/Validation",
                                            avg_loss,
                                            self.current_epoch)
         
        self.logger.experiment.add_scalar("Accuracy/Validation",
                                            correct/total,
                                            self.current_epoch)
 
        epoch_dictionary={
            # required
            'loss': avg_loss}
 
        return epoch_dictionary
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer



def main(model_name, 
         epoch_number,
         version_number):
    dataset = load_dataset("gsm8k","main")

    model = GPT2Finetuner(model_name)
    global tokenizer
    tokenizer = model.tokenizer

    random.seed(42)

    PROMPT_NEXT_SAMPL_TOKEN = model.tokenizer.eos_token+"\n"
    PROMPT_QASTION_TOKEN = "QA: "
    PROMPT_ANSWER_TOKEN = "A: "



    train_samples = dataset['train']
    # finetune_data = train_samples['test']
    finetune_data = train_samples

    finetune_dataset = TextGenerationDataset(finetune_data)
    # validation_data = dataset['test'].select(random.sample(list(range(len(dataset['test']))),len(dataset['test'])))
    validation_data = dataset['test']
    validation_dataset = TextGenerationDataset(validation_data)
    fintune_data_loader = DataLoader(dataset=finetune_dataset,
                                    batch_size=2,
                                    shuffle=True,
                                    collate_fn=finetune_collate_tokenizer)
    validation_data_loader = DataLoader(dataset=validation_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        collate_fn=finetune_collate_tokenizer)
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=version_number, name="lightning_logs")
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{model_name}',
                                          every_n_epochs=1)

    trainer = pl.Trainer(devices=2, 
                        accelerator="gpu",
                        logger=logger,
                        auto_lr_find=True,
                        max_epochs=epoch_number,
                        callbacks=[checkpoint_callback],
                        strategy="deepspeed")

    trainer.fit(model,
                train_dataloaders=fintune_data_loader,
                val_dataloaders=validation_data_loader)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="gpt2-large",
                        help="gpt2 models",
                        type=str)
    parser.add_argument("--epoch",
                        default=20,
                        help="maximux epoch number",
                        type=int)
    parser.add_argument("--version",
                        default=3, 
                        help="version for log",
                        type=int)

    args = parser.parse_args()
    main(args.model, 
         args.epoch,
         args.version)
       
