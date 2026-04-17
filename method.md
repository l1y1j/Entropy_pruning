# Method

## 3.1 Preliminary: Swin Transformer Background

Swin Transformer 是一种层次化的视觉骨干模型，其核心设计是基于窗口的滑动自注意力机制（Shifted Window MSA）。给定输入特征 $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C}$，Swin Transformer 通过非重叠的窗口划分将空间特征划分为 $N_{\text{win}} = \frac{H}{S_w} \times \frac{W}{S_w}$ 个窗口，其中 $S_w$ 为窗口大小（默认 $S_w = 7$）。每个窗口内的自注意力计算为：

$$\mathbf{A}^{(i)} = \text{Attention}\left(\mathbf{X}_w^{(i)} \mathbf{W}^Q, \mathbf{X}_w^{(i)} \mathbf{W}^K, \mathbf{X}_w^{(i)} \mathbf{W}^V\right)$$

其中 $i \in \{1, 2, \ldots, N_{\text{win}}\}$ 索引窗口，$\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ 为投影矩阵。标准 Swin Block 对所有窗口执行统一计算：

$$\mathbf{Y}_{\text{std}} = \text{FFN}(\text{LN}(\mathbf{A})) + \mathbf{X}_w$$

其中 $\text{LN}(\cdot)$ 为 Layer Normalization，$\text{FFN}(\cdot)$ 为前馈网络。

**问题**：标准 Swin 遵循"均匀计算假设"——所有窗口，无论其信息密度高低，都经历相同的注意力计算和 FFN 变换。这在信息高度稀疏的大规模场景中造成显著的计算浪费。

---

## 3.2 Overall Framework

针对均匀计算假设的局限性，我们提出**信息驱动的自适应计算分配框架**（Information-Driven Adaptive Computation Allocation）。该框架的核心洞察是：视觉信息冗余可以从两个互补的方面进行建模——**空间维度的非均匀性**（spatial non-uniformity）和**深度维度特征演化**（depth-wise feature evolution）。

具体而言：
- **空间非均匀性**：不同窗口的信息密度存在显著差异，部分窗口包含与全局背景显著不同的信息（通常是目标区域），而多数窗口为冗余背景
- **深度演化差异**：同一窗口在不同网络层的特征演化程度不同，部分窗口在早期层已达到良好表征，继续处理的信息增益有限

基于这一统一视角，我们设计**两阶段级联信息过滤框架**（Two-Stage Cascade Information Filtering）：

**第一阶段：空间显著性筛选。** 基于窗口级相对熵（KL Divergence），筛选高信息密度窗口执行完整注意力计算，低信息密度窗口以 Identity 方式跳过注意力。

**第二阶段：深度活跃度筛选。** 在第一阶段保留的高信息密度窗口上，进一步评估其在深度维度上的信息演化活跃程度。FFN 作为特征精炼模块（feature refinement module），对仍在经历非线性变换的窗口执行完整 FFN，对已达良好表征的窗口跳过 FFN。

两阶段筛选以级联方式依次执行，共同实现计算资源的自适应分配。

---

## 3.3 Stage 1: Spatial Saliency Filtering via KL Divergence

### 3.3.1 Motivation

在大规模场景图像中，信息分布高度不均匀。大部分区域为低信息密度的背景，其特征分布与全局背景分布高度相似——换言之，这些区域的特征可以被全局统计特性较好地预测。只有少数区域包含与全局背景显著不同的信息模式，这些区域往往对应于图像中的目标区域。

我们提出：**通过衡量每个窗口的特征分布与全局背景分布之间的偏离程度，来判断该窗口是否需要完整计算**。KL 散度（Relative Entropy）是量化这一偏离程度的理想工具。

### 3.3.2 Window-level Relative Entropy

**步骤1：计算局部分布**

对每个窗口内的特征取空间平均，得到该窗口的局部特征表示：

$$\boldsymbol{\mu}_i = \frac{1}{S_w^2}\sum_{j \in \mathcal{W}_i} \mathbf{X}_w^{(j)} \in \mathbb{R}^C$$

对该局部平均特征执行 Softmax，得到第 $i$ 个窗口的局部分布：

$$p_i^{(c)} = \frac{\exp(\mu_i^{(c)})}{\sum_{c'=1}^{C} \exp(\mu_i^{(c')})}, \quad \boldsymbol{p}_i \in \mathbb{R}^C$$

其中 $c$ 索引通道维度。

**步骤2：计算全局背景分布**

全局背景分布定义为所有窗口局部分布的均值：

$$\bar{p}^{(c)} = \frac{1}{N_{\text{win}}}\sum_{i=1}^{N_{\text{win}}}} p_i^{(c)}, \quad \bar{\boldsymbol{p}} \in \mathbb{R}^C$$

**步骤3：计算 KL 散度**

第 $i$ 个窗口相对于全局背景分布的 KL 散度定义为：

$$D_{\text{KL}}(i) = D_{\text{KL}}(\boldsymbol{p}_i \| \bar{\boldsymbol{p}}) = \sum_{c=1}^{C} p_i^{(c)} \log \frac{p_i^{(c)}}{\bar{p}^{(c)} + \epsilon} \tag{1}$$

其中 $\epsilon = 10^{-8}$ 用于数值稳定。

**物理意义**：$D_{\text{KL}}(i)$ 量化了第 $i$ 个窗口的局部分布与全局背景分布之间的信息偏离程度。高 KL 值意味着该窗口包含"不可预测"的信息——这类区域可能是目标所在；低 KL 值意味着该窗口的信息可由全局背景较好地推断，属于冗余背景区域。

> **Remark 1.** 与 Variance-based scoring（SparseFormer）相比，KL 散度具有更明确的信息论解释。Variance 仅衡量数值的离散程度，而 KL 散度衡量的是分布间的差异，直接对应"该区域与全局上下文的偏离程度"。

### 3.3.3 Top-K Selection Mechanism

基于 KL 散度得分，我们采用 Top-K 选择策略保留高信息密度窗口：

$$K = \left\lfloor N_{\text{win}} \times \rho_{\text{KL}} \right\rfloor$$

其中 $\rho_{\text{KL}} \in (0, 1]$ 为超参数，控制第一阶段的窗口保留比例。

设 $\mathcal{I}_{\text{keep}} = \text{TopK}\left(\{D_{\text{KL}}(i)\}_{i=1}^{N_{\text{win}}}, K\right)$ 为保留窗口的索引集合，$\mathcal{I}_{\text{bypass}} = \{1, 2, \ldots, N_{\text{win}}\} \setminus \mathcal{I}_{\text{keep}}$ 为跳过窗口的索引集合。筛选后的窗口特征定义为：

$$\mathbf{X}_{\text{attn}^{(i)}} = \begin{cases} \text{Attention}\left(\mathbf{X}_w^{(i)}\right), & \text{if } i \in \mathcal{I}_{\text{keep}} \\ \mathbf{X}_w^{(i)}, & \text{if } i \in \mathcal{I}_{\text{bypass}} \end{cases}$$

该机制使得低信息密度窗口直接透传原始特征，仅对高信息密度窗口执行注意力计算，从而在空间维度上实现计算资源的自适应分配。

---

## 3.4 Stage 2: Depth-wise Activity Filtering via Entropy Variation

### 3.4.1 Motivation

FFN（Feed-Forward Network）在 Transformer 中充当**特征精炼模块**（feature refinement module）：它通过非线性变换进一步精炼注意力层输出的特征表示。理想情况下，经过 FFN 处理后的特征应当更加鲁棒和具有判别性。

然而，并非所有窗口都需要 FFN 的精炼。对于某些窗口，其在注意力层输出的特征已经足够好——继续经过 FFN 的非线性变换带来的额外信息增益有限。对于另一些窗口，其特征仍在被深度提取，需要 FFN 进一步精炼。

如何判断一个窗口是否"需要 FFN 精炼"？我们提出：**通过衡量相邻网络层之间的特征熵变化来反映该窗口是否仍在经历非线性特征变换**。熵变化越大，说明特征仍处于活跃精炼状态，需要 FFN 继续处理；熵变化越小，说明特征已趋于稳定，已达到良好表征，可跳过 FFN。

> **Remark 2.** 该设计基于以下观察：FFN 的非线性变换会显著改变特征的通道分布，从而导致通道维度 Softmax 分布熵的变化。如果熵变化很小，说明特征分布已基本稳定，不需要 FFN 继续精炼。

### 3.4.2 Feature Entropy

给定第一阶段保留窗口的注意力输出 $\mathbf{A} \in \mathbb{R}^{N_{\text{keep}} \times S_w^2 \times C}$，我们首先计算每个窗口的特征熵。对通道维度执行 Softmax 得到注意力分布：

$$\tilde{a}_i^{(c)} = \frac{\exp(A_i^{(c)})}{\sum_{c'=1}^{C} \exp(A_i^{(c')})}$$

则第 $i$ 个窗口的特征熵定义为：

$$H_i = -\sum_{c=1}^{C} \tilde{a}_i^{(c)} \log\left(\tilde{a}_i^{(c)} + \epsilon\right) \tag{2}$$

其中 $\epsilon = 10^{-8}$ 用于数值稳定。$H_i$ 衡量第 $i$ 个窗口在注意力层输出后的特征复杂度。

### 3.4.3 Cross-layer Entropy Variation

我们维护一个熵缓存 $\mathbf{H}_{\text{cache}} \in \mathbb{R}^{N_{\text{win}}}$，初始化为零向量。跨层熵变化量定义为当前层熵与缓存熵的绝对差异：

$$\Delta H_i = \left| H_i - H_{\text{cache}}^{(i)} \right| \tag{3}$$

其中 $H_{\text{cache}}^{(i)}$ 为第 $i$ 个窗口在前一网络层的特征熵。

**物理意义**：$\Delta H_i$ 反映了第 $i$ 个窗口在相邻网络层间的特征演化程度：
- $\Delta H_i$ 较大 $\Rightarrow$ 该窗口的特征仍在被非线性变换精炼，处于活跃演化状态，需要继续 FFN 处理
- $\Delta H_i$ 较小 $\Rightarrow$ 该窗口的特征分布已趋于稳定，已达到良好表征，可跳过 FFN

> **Remark 3.** Stage 2 的筛选是**选择性的**：我们仅在高分辨率 Stage（如 Stage 2）中的特定 Block（如 Block 2, Block 4）上应用深度维度的活跃度筛选。这些位置的冗余最严重，因此信息精炼的收益也最显著。

### 3.4.4 Incremental Filtering

在第一阶段保留的窗口集合 $\mathcal{I}_{\text{keep}}$ 上，我们进一步执行增量筛选。设保留比例为 $\rho_{\Delta H} \in (0, 1]$，则进入 FFN 的窗口数为：

$$K' = \left\lfloor |\mathcal{I}_{\text{keep}}| \times \rho_{\Delta H} \right\rfloor$$

设 $\mathcal{J}_{\text{ffn}} = \text{TopK}\left(\{\Delta H_i\}_{i \in \mathcal{I}_{\text{keep}}}, K'\right)$ 为进入 FFN 的窗口索引，则 FFN 计算为：

$$\mathbf{Y}_{\text{ffn}^{(i)}} = \begin{cases} \text{FFN}\left(\text{LN}\left(\mathbf{A}^{(i)}\right)\right) + \mathbf{A}^{(i)}, & \text{if } i \in \mathcal{J}_{\text{ffn}} \\ \mathbf{A}^{(i)}, & \text{if } i \in \mathcal{I}_{\text{keep}} \setminus \mathcal{J}_{\text{ffn}} \end{cases}$$

同时更新熵缓存：

$$H_{\text{cache}}^{(i)} = \begin{cases} H\left(\mathbf{Y}_{\text{ffn}^{(i)}}\right), & \text{if } i \in \mathcal{J}_{\text{ffn}} \\ H_i, & \text{if } i \in \mathcal{I}_{\text{keep}} \setminus \mathcal{J}_{\text{ffn}} \end{cases}$$

---

## 3.5 Integration: SwinBlockEntropy Module

### 3.5.1 Module Architecture

我们将两阶段筛选机制集成到 Swin Transformer 的 Block 结构中，设计了 **SwinBlockEntropy** 模块。该模块继承标准 SwinBlock 的核心组件（Window MSA、FFN、DropPath），并在其 forward 过程中嵌入熵驱动的条件计算逻辑。

```python
class SwinBlockEntropy(SwinBlock):
    def forward(self, x, hw_shape, attn_entropy_cache=None):
        # Window partition with padding
        x_windows = self.window_partition(x)
        B, N_win = x_windows.shape[0] // B, x_windows.shape[1]
        
        if self.enable_entropy and self.shift_size == 0:
            # === Stage 1: KL-based Spatial Filtering ===
            kl_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
            k = max(1, int(N_win * self.kl_ratio))
            keep_idx = torch.topk(kl_scores, k=k, dim=1).indices.sort(dim=1).values
            
            # Partial attention on selected windows
            x_attn = self.partial_attention(x_windows, keep_idx)
            
            # === Stage 2: Entropy-based Depth Filtering ===
            if self.entropy_strategy == 'kl_incremental':
                cur_entropy = compute_attention_entropy(x_attn)
                inc_scores = torch.abs(cur_entropy - cached_entropy[keep_idx])
                k_ffn = max(1, int(len(keep_idx) * self.increment_ratio))
                ffn_idx = torch.topk(inc_scores, k=k_ffn, dim=1).indices.sort(dim=1).values
                
                # Partial FFN on high-entropy-variation windows
                x_ffn = self.partial_ffn(x_attn, ffn_idx)
                attn_entropy_cache[keep_idx[ffn_idx]] = compute_attention_entropy(x_ffn)
            else:
                x_ffn = self.full_ffn(x_attn)
                attn_entropy_cache[keep_idx] = compute_attention_entropy(x_ffn)
        else:
            # Standard Swin Block forward
            x_ffn = self.standard_forward(x_windows)
        
        # Window reverse and return
        return self.window_reverse(x_ffn, hw_shape), attn_entropy_cache
```

### 3.5.2 Multi-Mode Forward

SwinBlockEntropy 支持三种前向模式：

| 模式 | 描述 | 计算量 |
|------|------|--------|
| `standard` | 标准 Swin Block，无剪枝 | $1.0 \times$ |
| `kl` | 仅 Stage 1 KL 筛选 | $\rho_{\text{KL}} \times$ |
| `kl_incremental` | 两阶段级联筛选 | $\rho_{\text{KL}} \times \rho_{\Delta H} \times$ |

当 $\rho_{\text{KL}} = 1.0$ 且 $\rho_{\Delta H} = 1.0$ 时，SwinBlockEntropy 退化为标准 SwinBlock。

---

## 3.6 Computational Complexity Analysis

给定输入特征 $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C}$，标准 Swin Block 的计算复杂度为：

$$\mathcal{O}_{\text{std}} = \underbrace{B \cdot N_{\text{win}} \cdot S_w^2 \cdot C \cdot d}_{\text{Attention QKV}} + \underbrace{B \cdot N_{\text{win}} \cdot S_w^2 \cdot d \cdot C}_{\text{Attention Output}} + \underbrace{B \cdot N_{\text{win}} \cdot S_w^2 \cdot C \cdot 4C}_{\text{FFN}}$$

其中 $d$ 为注意力维度，$N_{\text{win}} = \frac{HW}{S_w^2}$。

本文方法的理论计算复杂度为：

$$\mathcal{O}_{\text{ours}} = B \cdot N_{\text{win}} \cdot \rho_{\text{KL}} \cdot \left[ S_w^2 \cdot C \cdot d + \rho_{\Delta H} \cdot \left(S_w^2 \cdot d \cdot C + S_w^2 \cdot C \cdot 4C\right) \right] \cdot \mathbf{1}_{\{\text{shift_size}=0\}} + B \cdot N_{\text{win}} \cdot (1 - \rho_{\text{KL}}) \cdot S_w^2 \cdot C$$

其中 $\mathbf{1}_{\{\text{shift_size}=0\}}$ 为指示函数（shifted window 不执行筛选）。

KL 散度和熵变化的计算开销为 $\mathcal{O}(B \cdot N_{\text{win}} \cdot S_w^2 \cdot C)$，相对于注意力计算的 $\mathcal{O}(B \cdot N_{\text{win}} \cdot S_w^2 \cdot d \cdot C)$ 可忽略不计，因为 $d \gg 1$。

> **Remark 4.** 实际加速比取决于输入图像的信息分布。在信息高度稀疏的大规模场景中，$\rho_{\text{KL}}$ 和 $\rho_{\Delta H}$ 可以设置得较低，从而获得显著加速；在信息分布均匀的标准数据集上，建议使用较高的保留比例以保持精度。

---

## 3.7 Unified Information-Theoretic Interpretation

为了完整性，我们对本方法的信息论基础进行统一阐释。

**空间维度（KL Divergence）**：KL 散度衡量的是第 $i$ 个窗口的局部分布 $\boldsymbol{p}_i$ 与全局背景分布 $\bar{\boldsymbol{p}}$ 之间的信息增益。这对应于"该窗口相对于全局背景的**空间信息冗余程度**"。

**深度维度（Entropy Variation）**：跨层熵变化量 $\Delta H_i = |H_i - H_{\text{cache}}^{(i)}|$ 衡量的是该窗口在相邻网络层间的特征演化程度。由于 FFN 充当特征精炼模块，熵变化反映的是"该窗口在当前网络深度上是否仍需要**非线性变换精炼**"。

**统一视角**：我们将两者统称为**信息冗余的双方面建模**（Dual-Aspect Information Redundancy Modeling）：
- **空间冗余**：某些窗口的信息可由全局背景预测（低 KL），属于空间上的冗余
- **深度冗余**：某些窗口的特征已趋于稳定（低 $\Delta H$），属于深度演化上的冗余

这两者共同决定了计算资源的分配策略：对空间冗余窗口跳过注意力，对深度冗余窗口跳过 FFN。

---

## 方法部分结构总结

| Section | 内容 | 核心公式 |
|---------|------|----------|
| 3.1 | Swin Background + 问题陈述 | - |
| 3.2 | Overall Framework + 统一视角 | Dual-Aspect Information Redundancy |
| 3.3.1 | KL Divergence 动机 | $D_{\text{KL}}(i) = \sum p_i \log(p_i / \bar{p}_i)$ |
| 3.3.2 | Top-K Selection | $K = \lfloor N_{\text{win}} \times \rho_{\text{KL}} \rfloor$ |
| 3.4.1 | FFN 功能解释 + Entropy 动机 | feature refinement module |
| 3.4.2 | Feature Entropy | $H_i = -\sum \tilde{a}_i \log \tilde{a}_i$ |
| 3.4.3 | Entropy Variation + selective application | $\Delta H_i = |H_i - H_{\text{cache}}^{(i)}|$ |
| 3.4.4 | Incremental Filtering | $K' = \lfloor |\mathcal{I}_{\text{keep}}| \times \rho_{\Delta H} \rfloor$ |
| 3.5 | SwinBlockEntropy 模块 | Algorithm + 代码 |
| 3.6 | 复杂度分析 | $\mathcal{O}_{\text{ours}}$ vs $\mathcal{O}_{\text{std}}$ |
| 3.7 | 统一信息论解释 | 空间冗余 + 深度冗余 |

---

## 尚需补充的视觉元素

根据 SparseFormer 的风格，方法部分还应包含以下 Figure：

| Figure | 描述 |
|--------|------|
| **Figure 1** | 两阶段筛选的整体框架图（两列：Stage 1 KL Filter + Stage 2 Entropy Filter） |
| **Figure 2** | KL Divergence 计算示意图（局部→全局→KL） |
| **Figure 3** | Entropy Variation 计算示意图（跨层熵缓存→差值→Top-K） |
| **Figure 4** | SwinBlockEntropy 模块结构图（与 SwinBlock 的对比） |
