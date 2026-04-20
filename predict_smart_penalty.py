import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ====================== 规则配置 ======================
ZODIAC = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]

COLOR_MAP = {
    '红': [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46],
    '蓝': [3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48],
    '绿': [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]
}

ELEMENT_MAP = {
    '金': [4,5,12,13,26,27,34,35,42,43],
    '木': [8,9,16,17,24,25,38,39,46,47],
    '水': [1,14,15,22,23,30,31,44,45],
    '火': [2,3,10,11,18,19,32,33,40,41,48,49],
    '土': [6,7,20,21,28,29,36,37]
}

DOMESTIC_WILD = {
    '家禽': ['马', '牛', '羊', '鸡', '狗', '猪'],
    '野兽': ['鼠', '虎', '兔', '龙', '蛇', '猴']
}

SIZE_MAP = {'小': list(range(1, 25)), '大': list(range(25, 49)), '49': [49]}
ODD_EVEN = {'单': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49],
            '双': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48]}
ZONE_MAP = {'1区': [1,2,3,4,5,6,7,8,9,10], '2区': [11,12,13,14,15,16,17,18,19,20],
            '3区': [21,22,23,24,25,26,27,28,29,30], '4区': [31,32,33,34,35,36,37,38,39,40],
            '5区': [41,42,43,44,45,46,47,48,49]}
TAIL_MAP = {'尾0': [10,20,30,40], '尾1': [1,11,21,31,41], '尾2': [2,12,22,32,42],
            '尾3': [3,13,23,33,43], '尾4': [4,14,24,34,44], '尾5': [5,15,25,35,45],
            '尾6': [6,16,26,36,46], '尾7': [7,17,27,37,47], '尾8': [8,18,28,38,48],
            '尾9': [9,19,29,39,49]}

traditional_map = {'龍': '龙', '馬': '马', '雞': '鸡', '豬': '猪'}

def create_reverse_map(mapping):
    return {v: k for k, values in mapping.items() for v in values}

color_by_num = create_reverse_map(COLOR_MAP)
element_by_num = create_reverse_map(ELEMENT_MAP)
size_by_num = create_reverse_map(SIZE_MAP)
oddeven_by_num = create_reverse_map(ODD_EVEN)
zone_by_num = create_reverse_map(ZONE_MAP)
tail_by_num = create_reverse_map(TAIL_MAP)

zodiac_encoding = {z: i for i, z in enumerate(ZODIAC)}
color_encoding = {'红': 0, '蓝': 1, '绿': 2}
element_encoding = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
size_encoding = {'小': 0, '大': 1, '49': 2}
oddeven_encoding = {'单': 0, '双': 1}
zone_encoding = {f'区{i}': i-1 for i in range(1, 6)}
tail_encoding = {f'尾{i}': i for i in range(10)}
domestic_encoding = {'家禽': 0, '野兽': 1}

def get_base_allocation():
    return {
        "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
        "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
        "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
        "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48]
    }

def zodiac_to_nums(zodiac, year=2026):
    base = get_base_allocation()
    return base[zodiac]

# ====================== 数据解析 ======================
data_str = """2026001 牛 29
2026002 猴 22
2026003 雞 09
2026004 雞 45
2026005 豬 43
2026006 蛇 13
2026007 牛 41
2026008 雞 21
2026009 虎 28
2026010 兔 27
2026011 羊 11
2026012 羊 11
2026013 蛇 01
2026014 龍 26
2026015 豬 31
2026016 鼠 06
2026017 狗 32
2026018 兔 39
2026019 猴 46
2026020 馬 12
2026021 鼠 42
2026022 鼠 18
2026023 羊 47
2026024 虎 28
2026025 蛇 37
2026026 豬 19
2026027 雞 45
2026028 兔 03
2026029 蛇 01
2026030 牛 41
2026031 龍 26
2026032 鼠 06
2026033 牛 41
2026034 豬 19
2026035 兔 27
2026036 蛇 01
2026037 豬 43
2026038 鼠 42
2026039 羊 11
2026040 虎 28
2026041 馬 36
2026042 虎 16
2026043 蛇 13
2026044 豬 19
2026045 蛇 01
2026046 兔 03
2026047 兔 39
2026048 羊 24
2026049 馬 13
2026050 猴 23
2026051 馬 01
2026052 蛇 38
2026053 龍 15
2026054 羊 48
2026055 蛇 14
2026056 羊 48
2026057 猴 23
2026058 鼠 31
2026059 雞 10
2026060 狗 09
2026061 虎 29
2026062 羊 24
2026063 兔 28
2026064 牛 18
2026065 虎 05
2026066 馬 37
2026067 虎 05
2026068 猴 23
2026069 羊 24
2026070 馬 25
2026071 羊 48
2026072 雞 46
2026073 雞 34
2026074 雞 10
2026075 狗 33
2026076 蛇 02
2026077 虎 29
2026078 雞 46
2026079 猴 35
2026080 龍 03
2026081 虎 17
2026082 龍 27
2026083 虎 05
2026084 兔 16
2026085 鼠 19
2026086 羊 12
2026087 蛇 26
2026088 龍 27
2026089 馬 49
2026090 羊 36
2026091 馬 37
2026092 牛 06
2026093 兔 40
2026094 虎 17
2026095 猴 35
2026096 鼠 43
2026097 猴 11
2026098 虎 17
2026099 羊 12
2026100 狗 33
2026101 龍 39
2026102 豬 20
2026103 牛 06
2026104 馬 01
2026105 兔 28
2026106 雞 22
2026107 馬 49
2026108 狗 45
2026109 兔 16"""

lines = data_str.strip().split('\n')
rows = []
for line in lines:
    p = line.split()
    issue, animal, num = p[0], p[1], int(p[2])
    animal = traditional_map.get(animal, animal)
    rows.append({
        '期号': issue, '生肖': animal, '号码': num,
        '颜色': color_by_num.get(num, '未知'),
        '五行': element_by_num.get(num, '未知'),
        '大小': size_by_num.get(num, '未知'),
        '单双': oddeven_by_num.get(num, '未知'),
        '区间': zone_by_num.get(num, '未知'),
        '尾数': tail_by_num.get(num, '未知'),
        '家禽野兽': '野兽' if animal in DOMESTIC_WILD['野兽'] else '家禽'
    })

df = pd.DataFrame(rows)
df['生肖_编码'] = df['生肖'].map(zodiac_encoding)
df['颜色_编码'] = df['颜色'].map(color_encoding)
df['五行_编码'] = df['五行'].map(element_encoding)
df['大小_编码'] = df['大小'].map(size_encoding)
df['单双_编码'] = df['单双'].map(oddeven_encoding)
df['区间_编码'] = df['区间'].map(zone_encoding)
df['尾数_编码'] = df['尾数'].map(tail_encoding)
df['家禽野兽_编码'] = df['家禽野兽'].map(domestic_encoding)

print(f"数据加载完成：共 {len(df)} 期")

# ====================== 分析历史模式 ======================
print("\n【分析历史模式】")

# 分析连续出现模式
consecutive_same = 0  # 连续相同生肖次数
consecutive_diff = 0   # 连续不同生肖次数
last_z = None

for z in df['生肖_编码'].values:
    if last_z is not None:
        if z == last_z:
            consecutive_same += 1
        else:
            consecutive_diff += 1
    last_z = z

total_consecutive = consecutive_same + consecutive_diff
same_prob = consecutive_same / total_consecutive if total_consecutive > 0 else 0
diff_prob = consecutive_diff / total_consecutive if total_consecutive > 0 else 0

print(f"连续相同生肖: {consecutive_same}次 ({same_prob:.1%})")
print(f"连续不同生肖: {consecutive_diff}次 ({diff_prob:.1%})")
print(f"结论: 连续{2 if same_prob > diff_prob else 1}个不同生肖的概率更高")

# 分析最近出现的生肖模式
print("\n最近10期生肖:")
recent_10 = df['生肖'].values[-10:]
print(" -> ".join(recent_10))

# ====================== 智能惩罚函数 ======================
def apply_smart_penalty(probs, last_two_zodiacs, consecutive_pattern):
    """
    智能惩罚函数：
    - 基于历史模式动态调整惩罚力度
    - 如果连续不同生肖更常见，降低上上期生肖概率
    - 如果连续相同生肖更常见，降低上期生肖概率
    """
    adjusted_probs = probs.copy()
    
    last_idx = last_two_zodiacs[0]
    second_last_idx = last_two_zodiacs[1]
    
    # 根据历史模式调整惩罚
    if consecutive_pattern['same_prob'] > consecutive_pattern['diff_prob']:
        # 连续相同更常见：主要惩罚上期
        adjusted_probs[last_idx] *= 0.3
        adjusted_probs[second_last_idx] *= 0.7
    else:
        # 连续不同更常见：主要惩罚上上期，降低连续相同概率
        adjusted_probs[last_idx] *= 0.6
        adjusted_probs[second_last_idx] *= 0.3
        
        # 相邻生肖提升（轮换规律）
        adjacent_indices = [
            (last_idx - 1) % 12,
            (last_idx + 1) % 12
        ]
        for adj_idx in adjacent_indices:
            if adj_idx != second_last_idx:  # 排除上上期
                adjusted_probs[adj_idx] *= 1.3
    
    # 归一化
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    return adjusted_probs

# 获取最近两期生肖
last_zodiac = df.iloc[-1]['生肖']
second_last_zodiac = df.iloc[-2]['生肖']
last_zodiac_idx = zodiac_encoding[last_zodiac]
second_last_zodiac_idx = zodiac_encoding[second_last_zodiac]

print(f"\n最近两期：{second_last_zodiac} -> {last_zodiac}")
print(f"连续模式：{'相同' if last_zodiac == second_last_zodiac else '不同'}")

consecutive_pattern = {
    'same_prob': same_prob,
    'diff_prob': diff_prob
}

# ====================== 综合特征工程 ======================
WINDOW = 16
TOTAL_DIM = 36 + 6 + 10 + 6 + 4 + 10 + 20 + 4 + 12  # 108维

def build_features(df, window):
    feats = []
    for i in range(len(df)):
        if i < window:
            feats.append(np.zeros(TOTAL_DIM))
        else:
            f = []
            recent = df.iloc[i-window:i]
            
            # 1. 生肖特征 - 36维
            for z in range(12):
                cnt = (recent['生肖_编码'] == z).sum()
                f.extend([cnt, cnt/window, window - list(recent['生肖_编码']).index(z) if z in recent['生肖_编码'].values else window])
            
            # 2. 颜色特征 - 6维
            for c in range(3):
                cnt = (recent['颜色_编码'] == c).sum()
                f.extend([cnt, cnt/window])
            
            # 3. 五行特征 - 10维
            for e in range(5):
                cnt = (recent['五行_编码'] == e).sum()
                f.extend([cnt, cnt/window])
            
            # 4. 大小特征 - 6维
            for s in range(3):
                cnt = (recent['大小_编码'] == s).sum()
                f.extend([cnt, cnt/window])
            
            # 5. 单双特征 - 4维
            for o in range(2):
                cnt = (recent['单双_编码'] == o).sum()
                f.extend([cnt, cnt/window])
            
            # 6. 区间特征 - 10维
            for z in range(5):
                cnt = (recent['区间_编码'] == z).sum()
                f.extend([cnt, cnt/window])
            
            # 7. 尾数特征 - 20维
            for t in range(10):
                cnt = (recent['尾数_编码'] == t).sum()
                f.extend([cnt, cnt/window])
            
            # 8. 家禽野兽特征 - 4维
            for d in range(2):
                cnt = (recent['家禽野兽_编码'] == d).sum()
                f.extend([cnt, cnt/window])
            
            # 9. 连出特征 - 12维
            zodiac_seq = recent['生肖_编码'].values
            for z in range(12):
                consecutive = 0
                current = 0
                for zv in zodiac_seq[::-1]:
                    if zv == z:
                        current += 1
                        consecutive = max(consecutive, current)
                    else:
                        current = 0
                f.append(consecutive)
            
            if len(f) < TOTAL_DIM:
                f.extend([0] * (TOTAL_DIM - len(f)))
            elif len(f) > TOTAL_DIM:
                f = f[:TOTAL_DIM]
            
            feats.append(np.array(f, dtype=float))
    return np.array(feats)

print("\n构建综合特征...")
stat_feats = build_features(df, WINDOW)
print(f"特征维度: {stat_feats.shape[1]}")

# 构造样本
n_samples = len(df) - WINDOW
X_seq = np.array([df['生肖_编码'].values[i:i+WINDOW] for i in range(n_samples)])
X_stat = stat_feats[WINDOW:]
y = df['生肖_编码'].values[WINDOW:]

scaler = StandardScaler()
X_stat = scaler.fit_transform(X_stat)

print(f"样本形状: X_seq={X_seq.shape}, X_stat={X_stat.shape}, y={y.shape}")

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ====================== 模型定义 ======================
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
    def call(self, x, training=False):
        att = self.att(x, x)
        x = self.ln1(x + self.drop1(att, training=training))
        return self.ln2(x + self.drop2(self.ffn(x), training=training))

class PositionalEncoding(layers.Layer):
    def __init__(self, pos, d_model):
        super().__init__()
        self.pos_emb = layers.Embedding(pos, d_model)
    def call(self, x):
        return x + self.pos_emb(tf.range(tf.shape(x)[1]))

def build_deep_model():
    d_model = 32
    seq_in = layers.Input(shape=(WINDOW,))
    x = layers.Embedding(12, d_model)(seq_in)
    x = PositionalEncoding(WINDOW, d_model)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = TransformerBlock(128, 2, 128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    stat_in = layers.Input(shape=(X_stat.shape[1],))
    s = layers.Dense(64, activation='relu')(stat_in)
    s = layers.Dense(32, activation='relu')(s)
    
    concat = layers.concatenate([x, s])
    concat = layers.Dense(64, activation='relu')(concat)
    out = layers.Dense(12, activation='softmax')(concat)
    
    model = keras.Model([seq_in, stat_in], out)
    
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====================== 训练模型 ======================
print("\n训练深度模型...")
final_deep = build_deep_model()
final_deep.fit(
    [X_seq, X_stat], 
    y, 
    epochs=10, 
    batch_size=32, 
    verbose=1,
    class_weight=class_weight_dict
)

print("\n训练LightGBM模型...")
final_lgb = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=12,
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
final_lgb.fit(X_stat, y)

# ====================== 历史验证 ======================
print("\n【历史验证 - 2026109期】")

prob_deep_109 = final_deep.predict([X_seq[-1:], X_stat[-1:]], verbose=0)[0]
prob_lgb_109 = final_lgb.predict_proba(X_stat[-1:])[0]
prob_original = 0.7 * prob_deep_109 + 0.3 * prob_lgb_109

print(f"最近两期：{second_last_zodiac} -> {last_zodiac}")
print(f"连续不同模式，惩罚策略：主要惩罚上上期{second_last_zodiac}")

# 应用智能惩罚
prob_adjusted = apply_smart_penalty(prob_original, [last_zodiac_idx, second_last_zodiac_idx], consecutive_pattern)

pred_df = pd.DataFrame({
    '生肖': ZODIAC,
    '原始概率': prob_original,
    '调整后概率': prob_adjusted
}).sort_values('调整后概率', ascending=False)

print(f"\n预测排序：")
for rank, (idx, row) in enumerate(pred_df.head(10).iterrows(), 1):
    nums = zodiac_to_nums(row['生肖'])
    print(f"TOP{rank} {row['生肖']}: {row['调整后概率']:.2%} (原:{row['原始概率']:.2%}) | 号码:{nums}")

actual_zodiac = df.iloc[-1]['生肖']
actual_num = df.iloc[-1]['号码']
print(f"\n实际结果: {actual_zodiac} {actual_num}")

if pred_df.iloc[0]['生肖'] == actual_zodiac:
    print("✅ TOP1 命中！")
elif actual_zodiac in pred_df.head(3)['生肖'].values:
    print("✅ TOP3 命中！")
else:
    print("❌ 未命中")

# ====================== 预测下一期 ======================
print("\n" + "="*60)
print("【预测下一期】")
print("="*60)

last_seq = df['生肖_编码'].values[-WINDOW:].reshape(1, WINDOW)
last_stat = stat_feats[-1].reshape(1, -1)
last_stat_scaled = scaler.transform(last_stat)

prob_deep_next = final_deep.predict([last_seq, last_stat_scaled], verbose=0)[0]
prob_lgb_next = final_lgb.predict_proba(last_stat_scaled)[0]
prob_next_original = 0.7 * prob_deep_next + 0.3 * prob_lgb_next

prob_next_adjusted = apply_smart_penalty(prob_next_original, [last_zodiac_idx, second_last_zodiac_idx], consecutive_pattern)

pred_df_next = pd.DataFrame({
    '生肖': ZODIAC,
    '原始概率': prob_next_original,
    '调整后概率': prob_next_adjusted
}).sort_values('调整后概率', ascending=False)

print(f"\n最近两期：{second_last_zodiac} -> {last_zodiac}")
print(f"预测下一期（连续不同后）：")
for rank, (idx, row) in enumerate(pred_df_next.head(10).iterrows(), 1):
    nums = zodiac_to_nums(row['生肖'])
    color = [color_by_num.get(n) for n in nums[:2]]
    elem = [element_by_num.get(n) for n in nums[:2]]
    print(f"TOP{rank} {row['生肖']}: {row['调整后概率']:.2%} | 号码:{nums} | 色:{color} | 五行:{elem}")

pred_df_next.to_csv('下一期预测_智能惩罚.csv', index=False, encoding='utf-8-sig')
print("\n已保存: 下一期预测_智能惩罚.csv")