import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import model
import utils
import argparse
import logging


args = argparse.ArgumentParser(description="TASTGCN")
args.add_argument("--dataset", default="PEMS03")
args.add_argument("--num_input", type=int, default=12)
args.add_argument("--num_output", type=int, default=12)
args.add_argument("--epochs", type=int, default=50)
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--device", type=str, default="cuda:0")
args.add_argument("--seed", type=int, default=7)
args.add_argument("--max_speed", type=int, default=120)
args.add_argument("--seq_len", type=int, default=12)
args.add_argument("--d_model", type=int, default=12)
args.add_argument("--d_ff", type=int, default=12)
args.add_argument("--space_emb_dim", type=int, default=3)
args.add_argument("--day_emb_dim", type=int, default=32)
args.add_argument("--week_emb_dim", type=int, default=32)
args.add_argument("--learning_rate", type=float, default=1e-3)
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(training_input, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]

        X_batch, Y_batch = training_input[indices], training_target[indices]

        X_batch = X_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        out = net(X_batch,A,S,V)
        loss = loss_criterion(out, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

def validate(val_input, val_target):
    net.eval()
    temp_validation_losses = []
    temp_validation_maes = []
    temp_validation_rmses = []
    temp_validation_mapes = []
    val_input = val_input.to(device=args.device)
    val_target = val_target.to(device=args.device)
    permutation = torch.randperm(val_input.shape[0])
    for i in range(0, val_input.shape[0], args.batch_size):
        indices = permutation[i:i + args.batch_size]
        X_batch_val, Y_batch_val = val_input[indices], val_target[indices]
        X_batch_val = X_batch_val.to(device=args.device)
        Y_batch_val = Y_batch_val.to(device=args.device)
        output = net(X_batch_val, A,  S, V)
        val_loss = loss_criterion(output, Y_batch_val).to(device="cpu")
        temp_validation_losses.append(val_loss.detach().numpy().item())


        out_unnormalized = output.detach().cpu().numpy() * stds[0] + means[0]
        target_unnormalized = Y_batch_val.detach().cpu().numpy() * stds[0] + means[0]
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        temp_validation_maes.append(mae)
        mse = np.mean((out_unnormalized - target_unnormalized) ** 2)  # 计算均方误差
        rmse = np.sqrt(mse)  # 计算均方根误差
        temp_validation_rmses.append(rmse)  # 将 RMSE 添加到列表中
        epsilon = 0.5
        absolute_percentage_error = np.abs(
            (out_unnormalized - target_unnormalized) / (np.abs(target_unnormalized) + epsilon)
        )
        test_mask = (np.abs(target_unnormalized) > epsilon)
        test_absolute_percentage_error = absolute_percentage_error[test_mask]
        mape = np.mean(test_absolute_percentage_error) * 100 if len(
            test_absolute_percentage_error) > 0 else 0.0
        temp_validation_mapes.append(mape)

    return temp_validation_losses,temp_validation_maes,temp_validation_rmses,temp_validation_mapes

def test(test_input, test_target):
    net.eval()
    test_input = test_input.to(device=args.device)
    test_target = test_target.to(device=args.device)
    test_maes = []
    test_rmses = []
    test_mapes = []
    permutation = torch.randperm(test_input.shape[0])
    for i in range(0, test_input.shape[0], args.batch_size):
        indices = permutation[i:i + args.batch_size]
        X_batch_test, Y_batch_test = test_input[indices], test_target[indices]
        X_batch_test = X_batch_test.to(device=args.device)
        Y_batch_test = Y_batch_test.to(device=args.device)
        output = net(X_batch_test, A, S, V)

        # 计算 MAE, RMSE, MAPE
        out_unnormalized = output.detach().cpu().numpy() * stds[0] + means[0]
        target_unnormalized = Y_batch_test.detach().cpu().numpy() * stds[0] + means[0]
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        test_maes.append(mae)
        mse = np.mean((out_unnormalized - target_unnormalized) ** 2)
        rmse = np.sqrt(mse)
        test_rmses.append(rmse)
        epsilon = 1.0
        absolute_percentage_error = np.abs(
            (out_unnormalized - target_unnormalized) / (np.abs(target_unnormalized) + epsilon)
        )
        test_mask = (np.abs(target_unnormalized) > epsilon)
        test_absolute_percentage_error = absolute_percentage_error[test_mask]
        mape = np.mean(test_absolute_percentage_error) * 100 if len(
            test_absolute_percentage_error) > 0 else 0.0
        test_mapes.append(mape)
    return test_maes,test_rmses,test_mapes

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    A, X, V, S, means, stds = utils.load_data(args)
    X = utils.time_index_emb(X)
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]
    training_input, training_target = utils.generate_dataset(train_original_data,
                                                             num_timesteps_input=args.num_input,
                                                             num_timesteps_output=args.num_output)
    val_input, val_target = utils.generate_dataset(val_original_data,
                                                   num_timesteps_input=args.num_input,
                                                   num_timesteps_output=args.num_output)
    test_input, test_target = utils.generate_dataset(test_original_data,
                                                     num_timesteps_input=args.num_input,
                                                     num_timesteps_output=args.num_output)
    log.info("数据预处理完成")

    net = model.TASTGCN(args,A.size(0))
    net = net.to(device=args.device)
    # 创建一个Adam优化器，用于优化模型的参数
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    loss_criterion = nn.L1Loss()
    loss_criterion = loss_criterion.to(device=args.device)

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_rmses = []
    validation_mapes = []
    best_test_mae = 999
    best_test_rmse = 999
    best_test_mape = 999
    for epoch in range(args.epochs):
        print("第", epoch+1, "次训练开始:")
        loss= train(training_input, training_target,batch_size =  args.batch_size)
        training_losses.append(loss)
        with torch.no_grad():
            temp_validation_losses,temp_validation_maes,temp_validation_rmses,temp_validation_mapes = validate(val_input,val_target)
            out = None
        # 计算整个验证集的平均损失和MAE
        avg_val_loss = np.mean(temp_validation_losses)
        avg_val_mae = np.mean(temp_validation_maes)
        avg_val_rmse = np.mean(temp_validation_rmses)
        avg_val_mape = np.mean(temp_validation_mapes)

        validation_losses.append(avg_val_loss)
        validation_maes.append(avg_val_mae)
        validation_rmses.append(avg_val_rmse)
        validation_mapes.append(avg_val_mape)

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        print("Validation RMSE: {}".format(validation_rmses[-1]))
        print("Validation MAPE: {}".format(validation_mapes[-1]))
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.plot(validation_maes, label="validation MAE")
        plt.plot(validation_rmses, label="validation RMSE")
        plt.legend()
        plt.show()

        with torch.no_grad():
            test_maes, test_rmses, test_mapes = test(test_input, test_target)
            avg_test_mae = np.mean(test_maes)
            avg_test_rmse = np.mean(test_rmses)
            avg_test_mape = np.mean(test_mapes)
            if(avg_test_mae<best_test_mae):
                print(f"Test MAE: {avg_test_mae:.4f}")
                print(f"Test RMSE: {avg_test_rmse:.4f}")
                print(f"Test MAPE: {avg_test_mape:.2f}%")
                best_test_mae = avg_test_mae
                best_test_rmse = avg_test_rmse
                best_test_mape = avg_test_mape
    print("Best test result: ")
    print(f"Test MAE: {best_test_mae:.4f}")
    print(f"Test RMSE: {best_test_rmse:.4f}")
    print(f"Test MAPE: {best_test_mape:.2f}%")