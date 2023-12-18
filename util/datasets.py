import  json, random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from PIL import Image
from util.base_prompt import build_prompt
from lavin import Tokenizer
import copy
import torch
import lightning as L

IMAGENET_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

class ScienceQADataSet(Data.Dataset):
    r"""A custom dataset class for ScienceQA data.

    Args:
        problems_path (str): Data path for 'problems.json'.
        pid_splits_path (str): Data path for 'pid_splits.json'.
        captions_path (str): Data path for 'captions.json'.
        images_path (str): Data path for images.
        tokenizer_path (str): Data path for 'tokenizer.model'.
        split (str): The data split to use (e.g., "train", "val").
        max_words (int, optional): Maximum number of words in a tokenized sequence. Default is 512.
        image_height (int, optional): Image height for input.
        image_width (int, optional): Image width for input.

    """
    def __init__(self, 
                 problems_path, 
                 pid_splits_path, 
                 captions_path, 
                 images_path, 
                 tokenizer_path, 
                 split, 
                 max_words,
                 image_height,
                 image_width,
                 prompt_format,
                 use_caption,
                 options
                ):
        super().__init__()

        self.problems = json.load(open(problems_path))
        pid_splits = json.load(open(pid_splits_path))
        captions = json.load(open(captions_path))["captions"]
        self.image_path=os.path.join(images_path,split)
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.max_words = max_words
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.prompt_format = prompt_format
        self.use_caption = use_caption
        self.options = options

        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]

        self.transforms = transforms.Compose([
            transforms.Resize((image_height, image_width), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self, prompt, answer):
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
        example_mask = example_mask
        label_mask = label_mask
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
        prompt_question,prompt_answer= build_prompt(self.problems, self.qids[idx], self.prompt_format, self.use_caption, self.options)

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3, self.image_height, self.image_width))
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

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

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
        example_mask = example_mask
        label_mask = label_mask
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
            image=torch.Tensor(torch.zeros(3,224,224))
            indicator=0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

class ScienceQADataModule(L.LightningDataModule):
    def __init__(self, 
                 problems_path: str = "./data/problems.json",
                 pid_splits_path: str = './data/pid_splits.json', 
                 captions_path: str = './data/captions.json', 
                 images_path: str = './data/images/', 
                 tokenizer_path: str = './data/weights/tokenizer.model',
                 max_words: int = 512,
                 image_height: int = 336,
                 image_width: int = 336,
                 train_batch_size: int = 1,
                 val_batch_size: int = 32,
                 prompt_format: str = 'QCM-ALE',
                 use_caption: bool = False,
                 options=["A", "B", "C", "D", "E"],
                 ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset_train = ScienceQADataSet(
                problems_path = self.hparams.problems_path, 
                pid_splits_path = self.hparams.pid_splits_path, 
                captions_path = self.hparams.captions_path, 
                images_path = self.hparams.images_path, 
                tokenizer_path = self.hparams.tokenizer_path, 
                split = 'train', 
                max_words = self.hparams.max_words,
                image_height = self.hparams.image_height,
                image_width = self.hparams.image_width,
                prompt_format = self.hparams.prompt_format,
                use_caption = self.hparams.use_caption,
                options = self.hparams.options
                )
            self.dataset_val = ScienceQADataSet(
                problems_path = self.hparams.problems_path, 
                pid_splits_path = self.hparams.pid_splits_path, 
                captions_path = self.hparams.captions_path, 
                images_path = self.hparams.images_path, 
                tokenizer_path = self.hparams.tokenizer_path, 
                split = 'val', 
                max_words = self.hparams.max_words,
                image_height = self.hparams.image_height,
                image_width = self.hparams.image_width,
                prompt_format = self.hparams.prompt_format,
                use_caption = self.hparams.use_caption,
                options = self.hparams.options
                )

    def train_dataloader(self):
        return Data.DataLoader(self.dataset_train, batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return Data.DataLoader(self.dataset_val, batch_size=self.hparams.val_batch_size)

if __name__ == '__main__':
    dataset = ScienceQADataModule()
    dataset.setup(stage="fit")
        
    count = 0
    for example, labels, example_mask, image, indicator in dataset.train_dataloader():
        count += 1
    print(F"Total count in training is {count}")
    print("An Example: ")
    print()
    print(example)
    print(labels)
    print(example_mask)
    print(indicator)

    print()
    print()
    dataset.setup(stage="val")
    
    count = 0
    for example, labels, example_mask, image, indicator in dataset.val_dataloader():
        count += 1
    print(F"Total count in validation is {count}")
    print("An Example: ")
    print()
    print(example)
    print(labels)
    print(example_mask)
    print(indicator)
