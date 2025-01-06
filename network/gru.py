import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 自定义实现的GRU
class CustomSimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(CustomSimpleGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建多层GRU的权重
        self.gru_layers = nn.ModuleList(
            [
                nn.Linear(
                    (
                        input_size + hidden_size
                        if layer == 0
                        else hidden_size + hidden_size
                    ),
                    3 * hidden_size,  # GRU只需要3个门（更新门、重置门和候选隐藏状态）
                )
                for layer in range(num_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Args:
            x: 输入张量, shape (batch_size, sequence_length, input_size)
            hidden: 初始隐藏状态列表 (可选)
        Returns:
            output: 预测结果
            h_n: 最终的隐藏状态
        """
        batch_size, seq_len, _ = x.size()

        # 初始化所有层的隐藏状态
        if hidden is None:
            h_prev = [
                torch.zeros(batch_size, self.hidden_size).to(x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_prev = hidden

        # 存储最后一层的所有时间步输出
        layer_outputs = []

        # 处理每个时间步
        for t in range(seq_len):
            # 当前输入
            layer_input = x[:, t, :]

            # 逐层处理
            for layer in range(self.num_layers):
                # 将输入和前一个隐藏状态连接
                combined = torch.cat([layer_input, h_prev[layer]], dim=1)

                # 计算所有门的值
                gates = self.gru_layers[layer](combined)
                update_gate, reset_gate, new_gate = gates.chunk(3, dim=1)

                # 应用激活函数
                update_gate = torch.sigmoid(update_gate)  # z_t
                reset_gate = torch.sigmoid(reset_gate)  # r_t

                # 计算候选隐藏状态
                reset_hidden = reset_gate * h_prev[layer]
                combined_reset = torch.cat([layer_input, reset_hidden], dim=1)
                new_gate = torch.tanh(new_gate)  # n_t

                # 更新隐藏状态 (使用凸组合)
                h_next = (1 - update_gate) * new_gate + update_gate * h_prev[layer]

                # 更新该层的状态
                h_prev[layer] = h_next

                # 当前层的输出作为下一层的输入
                layer_input = h_next

            # 保存最后一层的输出
            layer_outputs.append(h_next)

        # 将最后一层的所有输出堆叠起来
        outputs = torch.stack(layer_outputs)

        # 转换维度顺序为 (batch_size, seq_len, hidden_size)
        outputs = outputs.transpose(0, 1)

        # 通过全连接层
        predictions = self.fc(outputs)

        # 返回最后一个时间步的预测值
        return predictions[:, -1, :].squeeze()


# 2. 使用PyTorch内置GRU的简单实现
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 使用PyTorch的GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                x.device
            )

        # GRU前向传播
        out, hidden = self.gru(x, hidden)

        # 只需要最后一个时间步的输出
        out = self.fc(out[:, -1, :]).squeeze()
        return out


# 3. 创建数据集
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# 生成正弦波数据
t = np.linspace(0, 100, 1000)
data = np.sin(0.1 * t)
seq_length = 20

# 创建序列数据和目标值
X, y = create_sequences(data, seq_length)

# 转换为张量
X = torch.FloatTensor(X).unsqueeze(-1)  # 添加特征维度
y = torch.FloatTensor(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


if __name__ == "__main__":

    # 4. 训练模型
    # 设置超参数
    input_size = 1
    hidden_size = 32
    num_layers = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.01

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化模型、损失函数和优化器

    model = SimpleGRU(input_size, hidden_size, num_layers)
    # model = CustomSimpleGRU(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    train_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 5. 评估模型
    model.eval()
    with torch.no_grad():
        # 计算训练集得分
        train_predictions = model(X_train)
        train_mse = criterion(train_predictions, y_train)
        print(f"训练集 MSE: {train_mse.item():.4f}")

        # 计算测试集得分
        test_predictions = model(X_test)
        test_mse = criterion(test_predictions, y_test)
        print(f"测试集 MSE: {test_mse.item():.4f}")

    # 6. 可视化结果
    plt.figure(figsize=(15, 10))

    # 绘制训练损失
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 绘制预测结果
    plt.subplot(2, 1, 2)
    with torch.no_grad():
        # 获取所有预测值
        all_predictions = model(X).numpy()

    # 绘制真实值和预测值的比较
    plt.plot(y.numpy(), label="True Values", alpha=0.5)
    plt.plot(all_predictions, label="Predictions", alpha=0.5)
    plt.title("Predictions vs True Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig("gru.png")

    # 7. 使用模型进行预测
    def predict_next_n_values(model, initial_sequence, n_predictions):
        model.eval()
        current_sequence = initial_sequence.clone()
        predictions = []

        with torch.no_grad():
            for _ in range(n_predictions):
                # 获取预测值
                pred = model(current_sequence.unsqueeze(0))
                predictions.append(pred.item())

                # 更新序列
                current_sequence = torch.cat(
                    (current_sequence[1:], pred.reshape(1, 1)), dim=0
                )

        return predictions

    # 使用模型预测未来的值
    n_future = 100
    initial_sequence = X_test[0]  # 使用测试集的第一个序列作为初始序列
    future_predictions = predict_next_n_values(model, initial_sequence, n_future)

    # 绘制预测结果
    plt.figure(figsize=(15, 5))
    plt.plot(future_predictions, label="Predicted Future Values")
    plt.title("Future Value Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("future_gru.png")
