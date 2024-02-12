from bidict import bidict
import numpy as np
import torch
from typing import Tuple


class InductionHopsFullSequenceTask:
    def __init__(self, seq_len = 50, char_tokens = 5, min_hops = 0, max_hops = 3, rng: np.random.RandomState = None):
        self.seq_len = seq_len
        self.char_tokens = char_tokens
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.induction_tokens = max_hops - min_hops + 1
        self.num_tokens = char_tokens + self.induction_tokens + 2
        self.rng = rng

        self.BLANK_CHAR = '_'
        self.DOES_NOT_EXIST_CHAR = '~'

        MIN_CHAR_INT = 97
        MIN_NUM_INT = 48
        self.induction_token_map = bidict({chr(i + MIN_NUM_INT): i - min_hops for i in range(min_hops, max_hops + 1)})
        self.char_token_map = {chr(i): i - MIN_CHAR_INT + self.induction_tokens for i in range(MIN_CHAR_INT, MIN_CHAR_INT + char_tokens)}
        self.token_map = bidict({**self.induction_token_map, **self.char_token_map, **{self.BLANK_CHAR: self.num_tokens - 1, self.DOES_NOT_EXIST_CHAR: self.num_tokens - 2}})
        
    def get_batch(self, batch_size: int, metadata=False, hops=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensors = torch.zeros((batch_size, self.seq_len-1))
        output_tensors = torch.zeros((batch_size, self.seq_len-1))
        if metadata:
            input_strings = []
            output_strings = []
            induction_hop_strings_list = []
            induction_hop_indices_list = []
        for i in range(batch_size):
            if metadata:
                input_string, output_string, induction_hop_strings, induction_hop_indices = self.get_strings(metadata=True, hops=hops)
                input_strings.append(input_string)
                output_strings.append(output_string)
                induction_hop_strings_list.append(induction_hop_strings)
                induction_hop_indices_list.append(induction_hop_indices)
            else:
                input_string, output_string = self.get_strings(hops=hops)
            input_tensors[i] = torch.tensor([self.token_map[token] for token in input_string])
            output_tensors[i] = torch.tensor([self.token_map[token] for token in output_string])
        if metadata:
            return input_tensors.long(), output_tensors.long(), input_strings, output_strings, induction_hop_strings_list, induction_hop_indices_list
        else:
            return input_tensors.long(), output_tensors.long()

    def get_strings(self, metadata=False, hops=None):
        random_char_string = ''
        for i in range(self.seq_len):
            if i == 0:
                random_char_string += self.rng.choice(list(self.char_token_map.keys()))
            else:
                random_char_string += self.rng.choice(list(self.char_token_map.keys() - {random_char_string[i-1]}))

        induction_hop_strings = [random_char_string]
        induction_hop_indices = [range(self.seq_len)]
        for i in range(self.max_hops+1):
            last_string = induction_hop_strings[-1]
            last_indices = induction_hop_indices[-1]
            new_string = ''
            new_indices = []
            for i in range(self.seq_len):
                if last_indices[i] == -1 or (last_index := random_char_string[:last_indices[i]].rfind(last_string[i])) == -1:
                    new_string += self.DOES_NOT_EXIST_CHAR
                    new_indices.append(-1)
                else:
                    new_string += random_char_string[last_index + 1]
                    new_indices.append(last_index + 1)

            induction_hop_strings.append(new_string)
            induction_hop_indices.append(new_indices)

        if hops is None:
            num_hops = self.rng.randint(self.min_hops, self.max_hops+1)
        else:
            num_hops = hops

        input_string = self.induction_token_map.inv[num_hops - self.min_hops] + random_char_string[:-2]
        output_string = self.BLANK_CHAR + induction_hop_strings[num_hops][:-2]


        if metadata:
            return input_string, output_string, induction_hop_strings, induction_hop_indices
        else:
            return input_string, output_string


def get_task(task_name, rng: np.random.RandomState = None, **task_kwargs) -> InductionHopsFullSequenceTask:
    if task_name == "InductionHopsFullSequenceTask":
        return InductionHopsFullSequenceTask(rng=rng, **task_kwargs)