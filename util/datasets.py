import  json, re,random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from lavin import Tokenizer
import copy

IMAGENET_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

class ScienceQADataSet(Data.Dataset):
    r"""A custom dataset class for ScienceQA data.

    Args:
        args (namespace): Command line arguments containing data-related configurations.
        split (str): The data split to use (e.g., "train", "val", "test").
        model_path (str): Path to the model for tokenization.
        max_words (int, optional): Maximum number of words in a tokenized sequence. Default is 512.

    Attributes:
        args (namespace): Command line arguments containing data-related configurations.
        problems (dict): Raw data loaded from 'problems.json'.
        qids (list): List of question IDs for the specified split.
        image_path (str): Path to the image data for the specified split.
        tokenizer (Tokenizer): Tokenizer for processing text data.
        max_words (int): Maximum number of words in a tokenized sequence.
        split (str): The data split being used (e.g., "train", "val", "test").
        transforms (transforms.Compose): Image transformations applied to loaded images.
    """
    def __init__(self, args, split, model_path, max_words=512):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path=os.path.join(args.data_root,'images',split)
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((336, 336), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        r"""Tokenizes a prompt and an answer, and prepares them for model input.
        
        Args:
            prompt (str): The prompt text.
                Example:
                    'Context: N/A
                    Question: Which of these states is farthest north?
                    Options: (A) West Virginia (B) Louisiana (C) Arizona (D) Oklahoma
                    Response:'
            answer (str): The answer text.
                Example:
                    'The answer is A.'

        Returns:
            example (torch.Tensor): Tokenized and padded (with zero) input combining prompt and answer.
                Example:
                    tensor([    1, 15228, 29901,   405, 29914, 29909,    13, 16492, 29901,  8449,
                                310,  1438,  5922,   338,  2215,   386,   342,  6641, 29973,    13,
                                5856, 29901,   313, 29909, 29897,  3122, 11653,   313, 29933, 29897,
                                28838,   313, 29907, 29897, 23716,   313, 29928, 29897, 27879,    13,
                                5103, 29901,  1576,  1234,   338,   319, 29889,     2,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                    0,     0,     0,     0,     0,     0,     0,     0])
            labels (torch.Tensor): Tokenized labels with masked prompt in the beginning and padding at the end.
                Example:
                    tensor([    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,  1576,  1234,   338,   319, 29889,     2,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0])
            example_mask (torch.Tensor): Mask indicating valid tokens in the example.
                Example:
                    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0.])
            label_mask (torch.Tensor): Mask indicating valid tokens in the labels.
                Example:
                    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0.])
        """
        example=prompt+answer

        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1 # Masking question with -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask

    def __getitem__(self, idx):
        r"""Retrieve an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            example (torch.Tensor): Tokenized and formatted input combining prompt question and answer.
            labels (torch.Tensor): Tokenized labels with masking for prompt and padding.
            example_mask (torch.Tensor): Mask indicating valid tokens in the example.
            image (torch.Tensor): Processed image data if available, else a tensor of zeros.
            indicator (int): An indicator value (1 or 0) representing image availability.
        """
        prompt_question,prompt_answer= build_prompt(self.problems,self.qids[idx],self.args)

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,336,336).float())
            indicator=0

        example, labels, example_mask, _=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'all_data.json')))[split]

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        self.qids = [item['qid'] for item in self.data]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((336, 336), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question=self.data[idx]['instruction']
        prompt_answer=self.data[idx]['answer']

        if self.data[idx]['image'] is not None:
            # image_path='../data/images/train' if self.data[idx]['image_source']=='sqa' else '../data/images/train2014'
            if self.data[idx]['image_source'] == 'sqa':
                image = Image.open(os.path.join('../data/images/train', self.qids[idx], 'image.png')).convert('RGB')
            else:
                image = Image.open(os.path.join('../data/images/train2014',   'COCO_train2014_'+self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,336,336).float())
            indicator=0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.options = ["A", "B", "C", "D", "E"]
            self.use_caption = True
            self.prompt_format = 'CQM-A'
            self.data_root = '../data'
            self.output_root = '../output'
            self.caption_file = '../data/captions.json'
    cfg=Cfg()
    dataset=ScienceQADataSet(cfg,'val','../data/weights/')
    
    for example, labels, example_mask, image, indicator in dataset:
        print(example)
        print(labels)
