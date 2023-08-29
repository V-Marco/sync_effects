from data_preprocessing import line_to_tensor
from model import ClassRNN
import torch
import os, json, pickle
from data_preprocessing import n_letters

def predict(input_line, n_predictions = 3):

    with open("plist.json", "r") as file:
        plist = json.load(file)

    rnn = ClassRNN(n_letters, plist["n_hidden"], 18)
    rnn.load_state_dict(torch.load(os.path.join(plist["output_folder"], "weights")))
    rnn.eval()
    output = evaluate(rnn, line_to_tensor(input_line))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1)

    with open(os.path.join(plist["output_folder"], "all_categories.pickle"), "rb") as file:
        all_categories = pickle.load(file)

    predictions = []
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions


def evaluate(rnn, line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

if __name__ == "__main__":
    input_line = 'Petrov'
    predict(input_line)