from model import ClassRNN
from data_preprocessing import n_letters, read_data, line_to_tensor
import torch
import pandas as pd
import os, json, pickle
import matplotlib.pyplot as plt

def category_from_output(output, all_categories):
    _, top_idxs = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_idxs[0][0]
    return all_categories[category_i], category_i

import random

def random_training_pair(category_lines, all_categories):                                                                                                               
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)])
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

if __name__ == "__main__":

    with open("plist.json", "r") as file:
        plist = json.load(file)

    n_hidden = plist["n_hidden"]
    n_epochs = plist["n_epochs"]
    category_lines, all_categories, n_categories = read_data(plist["data_folder"])

    rnn = ClassRNN(n_letters, n_hidden, n_categories)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr = plist["lr"])

    all_loss = []
    current_loss = 0
    for epoch in range(1, n_epochs + 1):

        # Get random training input and target
        category, line, category_tensor, line_tensor = random_training_pair(category_lines, all_categories)

        rnn.zero_grad()
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()
        current_loss += float(loss.detach().numpy())

        if epoch % plist["plot_every"] == 0:
            avg_loss = current_loss / plist["plot_every"]
            print(f"[EPOCH {epoch}]: {avg_loss}")
            all_loss.append(avg_loss)
            current_loss = 0

    if not os.path.exists(plist["output_folder"]):
        os.mkdir(plist["output_folder"])

    torch.save(rnn.state_dict(), os.path.join(plist["output_folder"], "weights"))

    loss_df = pd.DataFrame({"loss": all_loss})
    loss_df.to_csv(os.path.join(plist["output_folder"], "loss.csv"), index = False)
    plt.plot(all_loss)
    plt.savefig(os.path.join(plist["output_folder"], "loss.png"))

    with open(os.path.join(plist["output_folder"], "all_categories.pickle"), "wb") as file:
        pickle.dump(all_categories, file)