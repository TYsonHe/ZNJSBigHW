# 粒子群优化算法（Particle Swarm Optimization, PSO）原理

## 1. 算法背景

粒子群优化算法（PSO）由Kennedy和Eberhart于1995年提出，是群体智能领域最具代表性的算法之一。该算法模拟鸟群觅食行为，通过个体间的信息共享来搜索最优解。

---

## 2. 数学模型

PSO将每个优化问题的解看作搜索空间中的一个"粒子"，每个粒子具有位置和速度两个属性，通过跟踪个体最优和群体最优来更新自身状态。

### 2.1 基本公式

**速度更新公式：**

$$v_i^{t+1} = w \cdot v_i^t + c_1 \cdot r_1 \cdot (p_{best,i} - x_i^t) + c_2 \cdot r_2 \cdot (g_{best} - x_i^t)$$

**位置更新公式：**

$$x_i^{t+1} = x_i^t + v_i^{t+1}$$

**参数说明：**
- $v_i^t$：第 $i$ 个粒子在第 $t$ 次迭代的速度
- $x_i^t$：第 $i$ 个粒子在第 $t$ 次迭代的位置
- $w$：惯性权重，控制粒子保持原有速度的程度
- $c_1$：个体学习因子（认知系数），调节粒子向自身历史最优位置靠近的步长
- $c_2$：社会学习因子（社会系数），调节粒子向全局最优位置靠近的步长
- $r_1, r_2$：$[0,1]$ 间的随机数
- $p_{best,i}$：第 $i$ 个粒子的历史最优位置
- $g_{best}$：整个群体的历史最优位置

### 2.2 惯性权重

**线性递减惯性权重（Shi & Eberhart, 1998）：**

$$w = w_{max} - \frac{w_{max} - w_{min}}{iter_{max}} \cdot t$$

其中 $w_{max} = 0.9$，$w_{min} = 0.4$，$t$ 为当前迭代次数。

**作用：**
- 前期 $w$ 较大，增强全局探索能力
- 后期 $w$ 较小，增强局部开发能力

---

## 3. 算法流程

```
输入：种群规模N，最大迭代次数Max_iter，搜索空间边界
输出：全局最优位置g_best及其适应度值

1. 随机初始化粒子位置和速度
2. 计算适应度值，初始化p_best和g_best
3. while t < Max_iter do
4.   更新惯性权重w
5.   for 每个粒子 i do
6.     更新速度 v_i
7.     更新位置 x_i
8.     越界处理
9.     计算适应度值
10.    若优于p_best_i则更新p_best_i
11.    若优于g_best则更新g_best
12.  end for
13.  t = t + 1
14. end while
15. return g_best
```

---

## 4. 算法特点

| 特点 | 说明 |
|------|------|
| **简单高效** | 仅需调整少数参数（w, c1, c2），实现简单 |
| **记忆能力强** | 通过p_best和g_best保留历史信息 |
| **全局信息共享** | 所有粒子共享g_best，信息传播速度快 |
| **收敛速度快** | 在单峰函数上收敛速度快 |

---

## 5. 算法不足

1. **易早熟收敛**：群体快速向g_best聚集，多样性丧失
2. **多峰函数表现差**：全局搜索能力不足，容易陷入局部最优
3. **参数敏感**：惯性权重和学习因子的选择对性能影响较大

---

## 参考文献

1. Kennedy J, Eberhart R. Particle swarm optimization[C]. Proceedings of ICNN'95 - International Conference on Neural Networks, 1995, 4: 1942-1948.
2. Shi Y, Eberhart R. A modified particle swarm optimizer[C]. Proceedings of IEEE International Conference on Evolutionary Computation, 1998: 69-73.
