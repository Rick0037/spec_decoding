import random
from collections import Counter


p = [0.6, 0.3, 0.1]  # target
q = [0.5, 0.4, 0.1]  # draft


def final_dist(p, q, resample):
    """接受路径 min(p, q) + 拒绝概率 * 拒绝后采样分布。"""

    overlap_list = []
    for i in range(len(p)):
        overlap_list.append(min(p[i], q[i]))
    beta = 1 - sum(overlap_list)  # reject
    return [o + beta * r for o, r in zip(overlap_list, resample)]


beta = sum(max(a - b, 0) for a, b in zip(p, q))
residual = [max(a - b, 0) / beta for a, b in zip(p, q)]
print("标准规则:", final_dist(p, q, residual))
print("错误规则 (拒绝后直接抽 p:)", final_dist(p, q, p))

# one-hot Draft：总是提议 q 的 argmax（索引 1，即第二个 Token），q 退化为 [0, 1, 0]
q_onehot = [0.0, 1.0, 0.0]
beta1 = sum(max(a - b, 0) for a, b in zip(p, q_onehot))
residual1 = [max(a - b, 0) / beta1 for a, b in zip(p, q_onehot)]
exact = final_dist(p, q_onehot, residual1)
assert all(abs(a - b) < 1e-12 for a, b in zip(exact, p))
print("one-hot Draft:", exact, "接受率:", 1 - beta1)

# 蒙特卡洛验证标准规则
rng = random.Random(42)
counts, accepted, trials = Counter(), 0, 100_000
for _ in range(trials):
    y = rng.choices(range(3), weights=q)[0]
    if rng.random() < min(1, p[y] / q[y]):
        token, accepted = y, accepted + 1
    else:
        token = rng.choices(range(3), weights=residual)[0]
    counts[token] += 1
print("经验分布:", [counts[i] / trials for i in range(3)],
      "接受率:", accepted / trials)
