import gymnasium as gym
import numpy as np
import os
import highway_env
import tensorflow as tf
import h5py

# 設定根目錄路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 自定義模型類別，用於手動加載權重
class CustomModel(tf.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([input_shape, 512], stddev=np.sqrt(2.0 / input_shape)), name="W1")
        self.b1 = tf.Variable(tf.zeros([512]), name="b1")
        self.W2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=np.sqrt(2.0 / 512)), name="W2")
        self.b2 = tf.Variable(tf.zeros([256]), name="b2")
        self.W3 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=np.sqrt(2.0 / 256)), name="W3")
        self.b3 = tf.Variable(tf.zeros([128]), name="b3")
        self.W4 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=np.sqrt(2.0 / 128)), name="W4")
        self.b4 = tf.Variable(tf.zeros([64]), name="b4")
        self.W5 = tf.Variable(tf.random.truncated_normal([64, 5], stddev=np.sqrt(2.0 / 64)), name="W5")
        self.b5 = tf.Variable(tf.zeros([5]), name="b5")

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.W2) + self.b2)
        x = tf.nn.relu(tf.matmul(x, self.W3) + self.b3)
        x = tf.nn.relu(tf.matmul(x, self.W4) + self.b4)
        return tf.nn.softmax(tf.matmul(x, self.W5) + self.b5)

def load_model_weights(model, model_path):
    # 加載權重文件並手動分配權重到模型
    with h5py.File(model_path, 'r') as f:
        model.W1.assign(f['W1'][:])
        model.b1.assign(f['b1'][:])
        model.W2.assign(f['W2'][:])
        model.b2.assign(f['b2'][:])
        model.W3.assign(f['W3'][:])
        model.b3.assign(f['b3'][:])
        model.W4.assign(f['W4'][:])
        model.b4.assign(f['b4'][:])
        model.W5.assign(f['W5'][:])
        model.b5.assign(f['b5'][:])

def load_validation_data():
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = valid_dataset['label']
    return valid_data, valid_label

def main():
    total_reward = 0

    # 手動構建模型並加載權重
    input_shape = 25  # 根據訓練時的輸入維度設置
    model = CustomModel(input_shape=input_shape)
    model_path = os.path.join(root_path, 'YOURMODEL.h5')
    load_model_weights(model, model_path)

    # 建立環境並檢查 roundabout-v0 是否可用
    try:
        env = gym.make('roundabout-v0', render_mode='rgb_array')
    except gym.error.NameNotFound:
        print("環境 'roundabout-v0' 不存在。請確認已正確安裝 highway-env。")
        return None

    # 加載驗證數據
    valid_data, valid_label = load_validation_data()
    correct_predictions = 0
    total_samples = 0

    for _ in range(10):  # 進行 10 輪測試
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            env.render()
            obs = obs.reshape(1, input_shape)  # 將觀察數據重塑為 (1, 25)
            
            # 預測動作
            logits = model(obs)  # 獲取原始 logits
            action = np.argmax(logits.numpy())  # 將 logits 轉換為動作

            # 比較預測結果與真實標籤
            true_label = valid_label[total_samples]  # 假設驗證數據與觀察對齊
            if action == true_label:
                correct_predictions += 1

            # 執行動作
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            total_samples += 1

    # 計算模型的準確率
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # 關閉環境
    env.close()

    # 輸出測試準確率和累計獎勵
    print(f"準確率: {accuracy * 100:.2f}%")
    print(f"10 輪後的總獎勵: {total_reward}")
    return total_reward

if __name__ == "__main__":
    main()  # 只執行一次 10 輪測試並結束
