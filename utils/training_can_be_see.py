import matplotlib.pyplot as plt
import re

# 读取日志文件
log_file = "/data/ljt/Test_7.15/logs/newunet_jx_new_metrics202507161451.log"
with open(log_file, 'r') as f:
    log_lines = f.readlines()

# 提取训练损失和验证损失
train_losses = []
val_losses = []
epochs = []

for line in log_lines:
    if "TRAIN" in line:
        # 提取训练损失
        match = re.search(r'Loss: (\d+\.\d+)', line)
        if match:
            train_losses.append(float(match.group(1)))
            # 提取epoch数
            epoch_match = re.search(r'TRAIN \((\d+)\)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
    elif "VAL" in line:
        # 提取验证损失
        match = re.search(r'Loss: (\d+\.\d+)', line)
        if match:
            val_losses.append(float(match.group(1)))

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', linewidth=2)
plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o', linewidth=2)

# 图表装饰
plt.title('Training and Validation Loss over Epochs', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

# 设置x轴刻度
plt.xticks(epochs[::5], rotation=45)  # 每5个epoch显示一个刻度

# 标记最低验证损失点
min_val_loss = min(val_losses)
min_idx = val_losses.index(min_val_loss)
plt.scatter(epochs[min_idx], min_val_loss, color='green', s=100, 
            label=f'Min Val Loss: {min_val_loss:.4f}')
plt.annotate(f'Min: {min_val_loss:.4f}\nEpoch: {epochs[min_idx]}', 
             xy=(epochs[min_idx], min_val_loss),
             xytext=(epochs[min_idx]+2, min_val_loss + 0.05),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->'))

# 调整布局
plt.tight_layout()

# 保存为图片
output_image = "loss_curve.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Loss curve saved to {output_image}")

# 显示图表（可选）
plt.show()