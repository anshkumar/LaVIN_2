from typing import List
import torch
from lavin.tokenizer import Tokenizer
from lavin.eval_model import Transformer

class LaVIN_Generator:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        images: torch.Tensor,
        indicators: List[int],
        max_gen_len: int,
        n_feats: int=3,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        prefix_img_token = self.tokenizer.encode("Image: ", bos=True, eos=False) # [BOS, ...]: [1, 7084, 29901, 29871]
        non_prefix_img_token= self.tokenizer.encode("Image: N/A", bos=True, eos=False) # [BOS, ...]: [1, 7084, 29901, 405, 29914, 29909]

        images=images.cuda()
        self.model.backbone.cuda()

        image_embeds= self.model.backbone.encode_image(images.half()).half() # [batch_size, num_feature, feature_dim]: [32, 6, 1024]
        image_embeds=self.model.adapter_proj(image_embeds) # [batch_size, num_feature, feature_dim]: [32, 6, 4096]

        prompt_tokens=[]
        for i, x in enumerate(prompts): # Iterate over a batch of questions and add image and non_image prefix to prompt. image_embeds can not be inserted directly here before applying positional embedding.
            if indicators[i]==1:
                # [1, 7084, 29901, 29871] + [894, 29901, ..., 29901]
                token_idx = prefix_img_token + self.tokenizer.encode(x, bos=False, eos=False)
            else:
                # [1, 7084, 29901, 405, 29914, 29909] + [894, 29901, ..., 29901]
                token_idx = non_prefix_img_token + self.tokenizer.encode(x, bos=False, eos=False)
            prompt_tokens.append(token_idx)

        min_prompt_size = min([len(t) for t in prompt_tokens])

        '''
        Convert variable length list of list to a fixed size Tensor with padded zeros.
        Example:
            Prompts:        [["Hello", "world"],
                             ["My", "name", "is", "Vedanshu"],
                             ["Capital", "of", "India", "is", "Delhi"]]

            input_text_mask:[[True, True, False, False, False, False],
                             [True, True, True, True, False, False],
                             [True, True, True, True, True, False]]
        '''
        tokens = torch.full((bsz, params.max_seq_len), 0).cuda().long()
        input_text_mask = torch.zeros_like(tokens).bool()
        for k, t in enumerate(prompt_tokens):
            t=t[:params.max_seq_len]
            tokens[k, :len(t)] = torch.tensor(t).long()
            input_text_mask[k, :len(t)]=True

        token_embeds=self.model.tok_embeddings(tokens) # [batch_size, seq_len] -> [batch_size, seq_len, feature_dim]: [32, 512, 4096]
        indicators=torch.Tensor(indicators).cuda().long()
        modality_embedding=self.model.adapter_modality_embedding(indicators).unsqueeze(1) #  [batch_size] -> [batch_size, 1, feature_dim]: [32, 1, 4096]

        for i in range(len(token_embeds)):
            if indicators[i] > 0:
                pos=len(prefix_img_token)
                # Insert image emebedding into the sequence here. Since, positional embedding have beedn inserted, image embedding can be inserted with the prompts.
                image_token_embed = torch.cat([token_embeds[i,:pos], image_embeds[i], token_embeds[i, pos:]],0)
                token_embeds[i] = image_token_embed[:params.max_seq_len]

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, params.max_seq_len):
            '''
            The prev_pos and cur_pos will like this:
                0, 2
                2, 3
                3, 4
                ...
            
                The loop will move min_prompt_size in first iteration, then will move by 1 in subsequent iterations. 
            '''

            if prev_pos==0:
                h = torch.cat([modality_embedding, token_embeds[:, prev_pos:cur_pos]], 1)
            else:
                h = token_embeds[:,prev_pos:cur_pos]
            logits = self.model.forward(h, prev_pos) # [batch_size, word_size]: [32, 32000]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1) # [batch_size, word_size]: [32, 32000]
                next_token = sample_top_p(probs, top_p) # [32, 1]
            else:
                next_token = torch.argmax(logits, dim=-1) # [32, 1]
            next_token = next_token.reshape(-1) # [32]

            # only replace token if prompt has already been generated
            '''
            For the Prompts:[["Hello", "world"],
                             ["My", "name", "is", "Vedanshu"],
                             ["Capital", "of", "India", "is", "Delhi"]]
            The output will be something like this:
                            [["Hello", "world", "!"],   <-- Next token here only in first iteration.
                             ["My", "name", "is", "Vedanshu"],  <-- Nothing will be changed here in first iteration.
                             ["Capital", "of", "India", "is", "Delhi"]]   <-- Nothing will be changed here in first iteration.
            '''
            next_token_embeds = torch.where(
                input_text_mask[:, cur_pos, None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token)
            ) # [batch_size] -> [batch_size, feature_dim]: [32, 4096]
            token_embeds[:,cur_pos]=next_token_embeds

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
