import unicodedata
import string
import os
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def read_data(path):

    # Build a list of names per language
    all_categories = []
    category_lines = {}
    
    for filename in os.listdir(path):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(os.path.join(path, filename))
        category_lines[category] = lines
    
    n_categories = len(all_categories)

    return category_lines, all_categories, n_categories


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters) # Mark, nonspacing

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# Turn a line into a <line_length x 1 x n_letters> or an array of one-hot letter vectors
def line_to_tensor(line, requires_grad = True):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    tensor.requires_grad = True
    return tensor