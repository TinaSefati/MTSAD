import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


losses_train = []  # To store the loss for each epoch
best_loss = 100
best_recall = 0
losses_test = []
recall_hist = []
precision_hist = []
f1_hist = []
auc_hist = []
TP_hist = []
TN_hist = []
FP_hist = []
FN_hist = []



# Define functions for each distinct phase

def train_epochs(epoch, num_epochs, model, optimizer, criterion, train_loader, device):
    total_loss = 0.0
    train_start_time = time.time()
    model.train()
    # Use num_epochs instead of NUM_EPOCHS
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tqdm_loader:
        for batch_data in tqdm_loader:
            optimizer.zero_grad()
            input_data, _ = batch_data  # Assuming data loader returns (input_data, labels)
            input_data = input_data.to(device)

            # Forward pass
            output_data = model(input_data)

            # Compute the loss
            loss = criterion(output_data, input_data)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update tqdm description with the current loss
            tqdm_loader.set_postfix({'Loss': loss.item()})

    training_time = time.time() - train_start_time
    return training_time, input_data, output_data


def plot_signals(input_data, output_data):
    # Plot real data
    plt.plot(input_data[-1].to('cpu').detach().numpy())
    plt.title("Real Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    # Plot reconstructed data
    plt.plot(output_data[-1].to('cpu').detach().numpy())
    plt.title("Reconstructed Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

def evaluate_epoch(epoch, num_epochs, model, test_loader, device, target_list):
    global losses_test, recall_hist, precision_hist, f1_hist, auc_hist
    global TP_hist, TN_hist, FP_hist, FN_hist

    anomaly_scores = []
    test_loss = 0.0
    model.eval()
    eval_start_time = time.time()

    with torch.no_grad():
        with tqdm(test_loader, desc="Computing Anomaly Scores", unit="batch") as tqdm_loader:
            for batch_data in tqdm_loader:
                input_data, _ = batch_data
                input_data = input_data.to(device)

                # Forward pass
                output_data = model(input_data)

                # Compute anomaly scores (MSE in this example)
                mse = nn.MSELoss(reduction='none')(output_data, input_data)
                mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
                anomaly_scores.extend(mse_per_sample.cpu().numpy())
                test_loss += np.average(anomaly_scores)

    avg_test_loss = test_loss / len(test_loader)
    losses_test.append(avg_test_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Test   Loss = {avg_test_loss}')

    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time

    # Plot histogram of anomaly scores
    anomaly_scores_np = np.array(anomaly_scores)
    plt.hist(anomaly_scores_np, bins=50, density=True, alpha=0.75)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.show()

    # Threshold optimization and evaluation
    anomaly_scores_tensor = torch.tensor(anomaly_scores_np, device='cpu')
    target_list = torch.tensor(target_list, device='cpu')

    percentiles = np.arange(90, 100, 0.5)
    f1_scores = []
    for percentile in percentiles:
        threshold = np.percentile(anomaly_scores_tensor, percentile)
        predictions = (anomaly_scores_tensor > threshold)
        f1 = f1_score(target_list, predictions, average='micro')
        f1_scores.append(f1)

    best_percentile = percentiles[np.argmax(f1_scores)]
    best_threshold = np.percentile(anomaly_scores_tensor, best_percentile)
    print(f'Best Threshold: {best_threshold:.4f} (at {best_percentile} percentile)')

    predicted_labels = (anomaly_scores_tensor > best_threshold)
    predicted_labels = np.array([-1 if x else 1 for x in predicted_labels])
    target_list = target_list.numpy()

    # Compute confusion matrix components
    TP = np.sum((target_list == -1) & (predicted_labels == -1))
    TN = np.sum((target_list == 1) & (predicted_labels == 1))
    FP = np.sum((target_list == 1) & (predicted_labels == -1))
    FN = np.sum((target_list == -1) & (predicted_labels == 1))


    # Calculate evaluation metrics
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1s_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(target_list, predicted_labels)

    recall_hist.append(recall)
    precision_hist.append(precision)
    f1_hist.append(f1s_score)
    auc_hist.append(auc)
    TP_hist.append(TP)
    TN_hist.append(TN)
    FP_hist.append(FP)
    FN_hist.append(FN)

    print(f'AUC: {auc:.8f}')
    print(f'Accuracy: {accuracy:.8f}')
    print(f'Precision: {precision:.8f}')
    print(f'Recall: {recall:.8f}')
    print(f'F1-Score: {f1s_score:.8f}')
    print('TP = ', TP)
    print('TN = ', TN)
    print('FP = ', FP)
    print('FN = ', FN)

    return eval_time

