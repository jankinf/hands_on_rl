import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 定义LSTM模型
class CustomSimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(CustomSimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建多层LSTM的权重
        self.lstm_layers = nn.ModuleList(
            [
                nn.Linear(
                    (
                        input_size + hidden_size
                        if layer == 0
                        else hidden_size + hidden_size
                    ),
                    4 * hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Args:
            x: 输入张量, shape (batch_size, sequence_length, input_size)
            hidden: 初始隐藏状态和细胞状态的元组列表 (可选)
        Returns:
            output: 预测结果
            (h_n, c_n): 最终的隐藏状态和细胞状态
        """
        batch_size, seq_len, _ = x.size()

        # 初始化所有层的隐藏状态
        if hidden is None:
            h_prev = [
                torch.zeros(batch_size, self.hidden_size).to(x.device)
                for _ in range(self.num_layers)
            ]
            c_prev = [
                torch.zeros(batch_size, self.hidden_size).to(x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_prev, c_prev = hidden

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
                gates = self.lstm_layers[layer](combined)
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)

                # 应用激活函数
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                c_gate = torch.tanh(c_gate)
                o_gate = torch.sigmoid(o_gate)

                # 更新细胞状态
                c_next = f_gate * c_prev[layer] + i_gate * c_gate

                # 计算新的隐藏状态
                h_next = o_gate * torch.tanh(c_next)

                # 更新该层的状态
                h_prev[layer] = h_next
                c_prev[layer] = c_next

                # 当前层的输出作为下一层的输入
                layer_input = h_next

            # 保存最后一层的输出
            layer_outputs.append(h_next)

        # 将最后一层的所有输出堆叠起来: (seq_len, batch_size, hidden_size)
        outputs = torch.stack(layer_outputs)

        # 转换维度顺序为 (batch_size, seq_len, hidden_size)
        outputs = outputs.transpose(0, 1)

        # 通过全连接层: (batch_size, seq_len, 1)
        predictions = self.fc(outputs)

        # 返回最后一个时间步的预测值和最终状态
        return predictions[:, -1, :].squeeze()


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # 初始化隐藏状态
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)

        # LSTM前向传播
        out, hidden = self.lstm(x, hidden)
        # 我们只需要最后一个时间步的输出
        out = self.fc(out[:, -1, :]).squeeze()
        return out


# 2. 创建数据集
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

    # 3. 训练模型
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

    # model = SimpleLSTM(input_size, hidden_size, num_layers)
    model = CustomSimpleLSTM(input_size, hidden_size, num_layers)
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

    # 4. 评估模型
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

    # 5. 可视化结果
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
    plt.savefig("lstm.png")

    # 6. 使用模型进行预测
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
    plt.savefig("future_lstm.png")
