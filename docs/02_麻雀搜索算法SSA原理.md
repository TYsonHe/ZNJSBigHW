# 麻雀搜索算法（Sparrow Search Algorithm, SSA）原理

## 1. 算法背景

麻雀搜索算法（SSA）由Xue和Shen于2020年提出，发表在《Systems Science & Control Engineering》期刊上。该算法受到麻雀群体觅食和反捕食行为的启发，是一种新型群体智能优化算法。

---

## 2. 行为模型

SSA将麻雀种群划分为三种角色：**发现者（Producer）**、**跟随者（Scrounger）** 和 **警戒者（Watchdog）**。

### 2.1 种群表示

麻雀位置矩阵：

$$X = \begin{bmatrix} x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\ x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n,1} & x_{n,2} & \cdots & x_{n,d} \end{bmatrix}$$

其中 $n$ 为麻雀数量，$d$ 为优化变量的维度。

适应度值向量：

$$F_X = \begin{bmatrix} f([x_{1,1} & x_{1,2} & \cdots & x_{1,d}]) \\ f([x_{2,1} & x_{2,2} & \cdots & x_{2,d}]) \\ \vdots \\ f([x_{n,1} & x_{n,2} & \cdots & x_{n,d}]) \end{bmatrix}$$

### 2.2 发现者（Producer）位置更新

发现者具有较高能量储备，负责搜索食物丰富的区域并引导整个种群移动。发现者占种群比例约20%。

**位置更新公式：**

$$X_{i,j}^{t+1} = \begin{cases} X_{i,j}^t \cdot \exp\left(-\frac{i}{\alpha \cdot iter_{max}}\right), & R_2 < ST \\ X_{i,j}^t + Q \cdot L, & R_2 \geq ST \end{cases}$$

**参数说明：**
- $t$：当前迭代次数
- $j = 1, 2, \ldots, d$：维度索引
- $iter_{max}$：最大迭代次数
- $\alpha \in (0, 1]$：随机数
- $R_2 \in [0, 1]$：报警值
- $ST \in [0.5, 1.0]$：安全阈值
- $Q$：服从正态分布的随机数
- $L$：各元素均为1的 $1 \times d$ 矩阵

**公式解读：**
- $R_2 < ST$：周围没有捕食者，发现者进入广泛搜索模式
- $R_2 \geq ST$：发现捕食者，所有麻雀需飞向安全区域

### 2.3 跟随者（Scrounger）位置更新

跟随者监视发现者，一旦发现者找到好的食物源，便立即离开当前位置竞争食物。

**位置更新公式：**

$$X_{i,j}^{t+1} = \begin{cases} Q \cdot \exp\left(\frac{X_{worst}^t - X_{i,j}^t}{i^2}\right), & i > n/2 \\ X_P^{t+1} + |X_{i,j}^t - X_P^{t+1}| \cdot A^+ \cdot L, & i \leq n/2 \end{cases}$$

**参数说明：**
- $X_P$：发现者占据的最优位置
- $X_{worst}$：当前全局最差位置
- $A$：各元素随机赋值为1或-1的 $1 \times d$ 矩阵
- $A^+ = A^T(AA^T)^{-1}$

**公式解读：**
- $i > n/2$：适应度值较差的跟随者可能饥饿，需飞往其他地方觅食
- $i \leq n/2$：跟随者在发现者附近随机搜索

### 2.4 警戒者（Watchdog）位置更新

感知危险的麻雀占总种群的10%~20%，初始位置在种群中随机生成。

**位置更新公式：**

$$X_{i,j}^{t+1} = \begin{cases} X_{best}^t + \beta \cdot |X_{i,j}^t - X_{best}^t|, & f_i > f_g \\ X_{i,j}^t + K \cdot \left(\frac{|X_{i,j}^t - X_{worst}^t|}{(f_i - f_w) + \varepsilon}\right), & f_i = f_g \end{cases}$$

**参数说明：**
- $X_{best}$：当前全局最优位置
- $\beta$：步长控制参数，服从均值为0、方差为1的正态分布
- $K \in [-1, 1]$：随机数，表示麻雀移动方向和步长
- $f_i$：当前麻雀的适应度值
- $f_g$、$f_w$：当前全局最优和最差适应度值
- $\varepsilon$：最小常数，避免除零错误

**公式解读：**
- $f_i > f_g$：麻雀处于种群边缘，向安全区域（最优位置）靠近
- $f_i = f_g$：麻雀处于种群中间，意识到危险需向其他个体靠拢

---

## 3. 算法流程

```
输入：最大迭代次数G，发现者数量PD，警戒者数量SD，报警值R2，麻雀数量n
输出：全局最优位置X_best及其适应度值f_g

1. 初始化n只麻雀的位置和相关参数
2. while t < G do
3.   计算适应度值，排序并记录当前最优和最差个体
4.   R2 = rand(1)
5.   for i = 1:PD do
6.     用公式(3)更新发现者位置
7.   end for
8.   for i = (PD+1):n do
9.     用公式(4)更新跟随者位置
10.  end for
11.  for l = 1:SD do
12.    用公式(5)更新警戒者位置
13.  end for
14.  获取当前新位置
15.  若新位置优于旧位置则更新
16.  t = t + 1
17. end while
18. return X_best, f_g
```

---

## 4. 算法特点

| 特点 | 说明 |
|------|------|
| **角色划分明确** | 发现者、跟随者、警戒者三种角色各司其职 |
| **探索与开发分离** | 发现者负责全局探索，跟随者负责局部开发 |
| **警报机制** | 通过R2和ST参数引入动态警报机制，增强算法灵活性 |
| **自适应搜索** | 发现者的搜索范围随迭代次数自适应调整 |

---

## 5. 算法不足

1. **易陷入局部最优**：发现者的搜索模式在后期可能过于集中
2. **参数敏感**：安全阈值ST和报警值R2的设定对算法性能影响较大
3. **跟随者多样性不足**：跟随者主要围绕发现者搜索，可能导致早熟收敛

---

## 参考文献

Xue J, Shen B. A novel swarm intelligence optimization approach: sparrow search algorithm[J]. Systems Science & Control Engineering, 2020, 8(1): 22-34.
