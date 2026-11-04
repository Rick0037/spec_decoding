import random
from collections import Counter


def decoding_receive_test():
    p = [0.6, 0.3, 0.1]  # target
    q = [0.2, 0.5, 0.3]  # draft

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


def suffix_match_test():
    def propose(history, min_n=2, max_n=4, k=3):
        """返回 (命中窗口长度, 候选)；未命中返回 (None, [])。"""
        for n in range(min(max_n, len(history) - 1), min_n - 1, -1):
            tail = history[-n:]
            # 起点 < len - n，排除尾巴自身；从前往后扫，取最早的匹配
            for start in range(len(history) - n):
                if history[start:start + n] == tail:
                    return n, history[start + n:start + n + k]
        return None, []

    h1 = [10, 11, 12, 20, 21, 99, 10, 11, 12]
    print("h1:", propose(h1))                      # (3, [20, 21, 99])
    print("h1 只允许 4-gram:", propose(h1, min_n=4))  # (None, [])

    h2 = [7, 8, 1, 7, 8, 2, 7, 8]                  # 尾巴 [7, 8] 出现过两次
    print("h2:", propose(h2))                      # 取最早一处：(2, [1, 7, 8])
    print("没有重复:", propose([1, 2, 3, 4, 5]))


# def __main__() {
#     suffix_match_test()
#     return
# }


def decoding_test():
    parents = [-1, 0, 0, 1, 2]
    n = len(parents)
    mask = [[0] * n for _ in range(n)]
    depth = [0] * n

    for node, parent in enumerate(parents):
        assert parent < node, "要求父节点排在子节点之前"
        depth[node] = 0 if parent == -1 else depth[parent] + 1
        cursor = node
        while cursor != -1:          # 沿祖先链往上，逐个标记可见
            mask[node][cursor] = 1
            cursor = parents[cursor]

    causal = [[1 if j <= i else 0 for j in range(n)] for i in range(n)]
    print("相对 Position ID:", depth)
    for i, (tree_row, causal_row) in enumerate(zip(mask, causal)):
        leaked = [j for j in range(n) if causal_row[j] and not tree_row[j]]
        print(f"节点 {i}: 树形 Mask {tree_row}  下三角会多看到 {leaked}")

    assert mask[4] == [1, 0, 1, 0, 1]
    assert depth[1] == depth[2] == 1

    # 相对 Position ID: [0, 1, 1, 2, 2]
    # 节点 0: 树形 Mask [1, 0, 0, 0, 0]  下三角会多看到 []
    # 节点 1: 树形 Mask [1, 1, 0, 0, 0]  下三角会多看到 []
    # 节点 2: 树形 Mask [1, 0, 1, 0, 0]  下三角会多看到 [1]
    # 节点 3: 树形 Mask [1, 1, 0, 1, 0]  下三角会多看到 [2]
    # 节点 4: 树形 Mask [1, 0, 1, 0, 1]  下三角会多看到 [1, 3]


if __name__ == "__main__":
    if True:
        decoding_test()
        # suffix_match_test()
    else:
        decoding_receive_test()
