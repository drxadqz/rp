# C3-FaRNet 当前最佳算法中文说明

本文档说明仓库中已经完整训练并在 RSCD 全量测试集上验证过的当前最佳单模型：

```text
c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709
```

该模型不是多模型集成，也不是测试时投票结果。它是一个单模型结果，完整测试集包含 49,500 张图像。

## 1. 任务定义

RSCD 不是普通的 27 类平面分类问题。每个类别本质上由若干个物理和视觉因素组合而成，例如：

```text
water_concrete_slight = water + concrete + slight
```

也就是说，模型需要同时判断：

- 路面状态或低摩擦线索，例如 dry、wet、water、snow、ice；
- 路面材质，例如 asphalt、concrete、mud、gravel；
- 路面粗糙程度，例如 smooth、slight、severe。

当前算法的核心思想是：不要把 27 个类别当作彼此独立的名字死记硬背，而是显式学习这些因素以及因素之间的耦合关系。

## 2. 输入与数据协议

输入图像统一处理为：

```text
192 x 192
```

预处理采用 letterbox resize，即保持原图宽高比，再用 padding 补齐到固定尺寸。这样做的原因是 RSCD 中很多图像是小路面 patch，如果直接拉伸成正方形，纹理方向、裂纹形状和粗糙度线索会被改变。

仓库使用 CSV manifest 管理数据，主要字段包括：

```text
image_path, split, dataset, class_label,
friction_label, material_label, unevenness_label,
wetness_label, snow_label, risk_label, mu_low, mu_high
```

其中 `class_label` 是 27 类组合标签；`friction_label`、`material_label` 和 `unevenness_label` 是从组合标签解析出来的因素标签。

## 3. 标签因子分解

算法把每个类别写成三元组：

```text
y = (f, m, r)
```

其中：

- `f` 表示 friction/condition factor，即路面状态或低摩擦因素；
- `m` 表示 material factor，即路面材质；
- `r` 表示 roughness factor，即路面粗糙程度。

如果只使用普通分类头，模型学习的是：

```text
p(y | x)
```

而 C3-FaRNet 同时关心：

```text
p(f | x), p(m | x), p(r | x), p(f, m, r | x)
```

这样做的好处是，当 `water_concrete_slight` 和 `wet_concrete_slight` 混淆时，模型知道这两个类别主要差在 `water` 和 `wet` 的状态边界，而不是完全无结构地重新学习两个类别。

## 4. 总体架构

当前最佳单模型由五个主要部分组成：

```text
image
  -> ConvNeXt visual backbone
  -> PhysicsTexture
  -> LocalPhysicsField
  -> SemanticPhysicsAttention
  -> coupled factor / hard-pair calibrated head
  -> 27-class logits
```

其中 ConvNeXt 负责提取通用视觉特征；PhysicsTexture、LocalPhysicsField 和 SemanticPhysicsAttention 负责提供与路面摩擦可供性相关的物理视觉证据；最终分类头负责把全局视觉特征和物理证据融合成 27 类输出。

当前验证模型参数量为：

```text
32.49M total parameters
1.09M trainable parameters under the S7 prefix-tuning setup
```

## 5. PhysicsTexture 模块

PhysicsTexture 是当前算法中最关键的显式物理视觉分支。它不直接预测类别，而是从图像中计算与摩擦、湿滑、积水、雪冰、粗糙度相关的可解释线索。

首先把 RGB 图像转为灰度图：

```text
g = 0.299 R + 0.587 G + 0.114 B
```

其中：

- `R, G, B` 是红、绿、蓝三个颜色通道；
- `g` 是亮度图；
- 系数 0.299、0.587、0.114 来自常用亮度转换公式，表示人眼对绿色最敏感，对蓝色相对不敏感。

颜色饱和度近似为：

```text
s = (max(R, G, B) - min(R, G, B)) / (max(R, G, B) + eps)
```

其中：

- `s` 越大，颜色越鲜明；
- `s` 越小，图像越接近灰白或灰黑；
- `eps` 是很小的数，用来避免分母为 0。

湿滑证据可以概括为：

```text
E_wet = clip(E_specular + 0.5 E_dark-water, 0, 1)
```

其中：

- `E_specular` 表示镜面高光，常见于湿路面或水膜；
- `E_dark-water` 表示暗色平滑积水区域；
- `clip` 把结果限制在 0 到 1 之间。

粗糙度证据可以概括为：

```text
E_rough = sigmoid((||grad g||_2 + 0.5 |lap(g)| - 0.12) * 12)
```

其中：

- `grad g` 是灰度图的一阶梯度，用来表示边缘和纹理强度；
- `lap(g)` 是 Laplacian 响应，用来表示高频细节和局部突变；
- `sigmoid` 把证据压缩到 0 到 1；
- 粗糙路面的砂石、裂纹和颗粒通常会带来更强的梯度和 Laplacian 响应。

大白话理解：PhysicsTexture 就像先让模型单独看“路面是不是反光”“有没有水膜”“纹理有没有被水抹平”“砂石裂纹是否明显”，再把这些线索交给主模型。

实现位置：

```text
src/friction_affordance/models/texture.py
```

## 6. LocalPhysicsField 模块

如果只看整张图的平均值，局部水膜、局部反光或局部粗糙区域容易被淹没。LocalPhysicsField 因此把物理线索做成局部场：

```text
L_eta(x)
```

其中：

- `x` 是输入图像；
- `eta` 表示该模块的参数；
- `L_eta(x)` 表示局部物理证据特征。

它重点观察局部区域中的：

- 反光水膜；
- 暗色积水；
- 纹理擦除；
- 局部粗糙边缘；
- 局部对比度。

大白话理解：PhysicsTexture 是“整张图有哪些物理线索”，LocalPhysicsField 是“这些线索具体出现在图像哪里、范围多大、是否集中”。

## 7. SemanticPhysicsAttention 模块

不同类别需要看的证据不同。比如：

- wet/water 更应该看水膜、反光、暗水区域；
- concrete 更应该看水泥纹理和灰度分布；
- slight/severe 更应该看颗粒、裂纹和粗糙变化；
- snow/ice 更应该看白度、低饱和度和纹理丢失。

SemanticPhysicsAttention 学习一个类别相关的物理证据注意力：

```text
A_psi(x)
```

其中：

- `psi` 是注意力模块参数；
- `A_psi(x)` 是语义相关的物理证据表示。

大白话理解：它不是把所有物理线索平均混在一起，而是让每一类去看自己最需要的证据。

## 8. 耦合因子分类头

最简单的因子模型可以写成：

```text
Z(f, m, r) = A_f + B_m + C_r
```

其中：

- `Z(f, m, r)` 是组合类别 `(f, m, r)` 的 logit；
- `A_f` 是状态因素得分；
- `B_m` 是材质因素得分；
- `C_r` 是粗糙度因素得分。

但 RSCD 的难点在于，不同因素不是简单相加。比如 `wet + concrete + slight` 的视觉表现，并不等于 wet、concrete 和 slight 三个单独特征的简单相加。因此当前算法使用更完整的耦合形式：

```text
Z(f, m, r) = A_f + B_m + C_r + D_fm + E_fr + G_mr + H_fmr
```

其中：

- `D_fm` 表示状态和材质之间的二阶耦合；
- `E_fr` 表示状态和粗糙度之间的二阶耦合；
- `G_mr` 表示材质和粗糙度之间的二阶耦合；
- `H_fmr` 表示状态、材质、粗糙度三者同时出现时的三阶耦合。

大白话理解：水在沥青上、水在水泥上、雪在水泥上、湿滑水泥轻微粗糙，这些组合不是一个统一公式能完全描述的，所以模型需要显式学习“组合之后长什么样”。

## 9. Hard-Pair Error-Gated Calibration

当前最佳模型使用：

```text
head_type = hardpair_error_gated_calibrated
```

Hard pair 指只差一个因素的类别对，例如：

```text
water_concrete_slight <-> wet_concrete_slight
dry_concrete_smooth <-> dry_concrete_slight
```

这类边界非常容易混淆。该模块做的事情是：

1. 先用主分类器得到 27 类 logits；
2. 判断样本是否靠近某个已知 hard-pair 边界；
3. 如果确实靠近边界，就允许专门的 hard-pair expert 做小幅校准；
4. 如果不靠近边界，就尽量不动原预测。

大白话理解：它不是把所有类别都重调一遍，而是只在最容易错的边界附近小心修正，避免“修好一个类、弄坏另一个类”。

## 10. Source-Reliable Boundary Router

当前 S7 模型还启用了：

```text
use_source_reliable_boundary_router = true
```

它的作用是：当一个源类别比较可靠，并且目标类别是预定义的邻近边界类别时，模型可以在物理证据支持的情况下，把一部分 logit 从源类别转移到目标类别。

当前公开配置中的有效路由包括：

```text
dry_concrete_smooth -> dry_concrete_slight
```

这主要用于修正 dry concrete 中 smooth 和 slight 的细粒度粗糙度边界。

## 11. 损失函数

主损失是 27 类交叉熵：

```text
L_CE = - log p_y
```

其中：

- `p_y` 是模型给真实类别 `y` 的预测概率；
- 交叉熵越小，说明模型越相信正确类别。

训练时还使用 anchor consistency 和 no-flip 保护项。简化写法为：

```text
L = L_CE + lambda_h L_anchor + lambda_r L_relation
```

其中：

- `L_anchor` 约束新模型不要随意偏离已经较好的旧模型；
- `L_relation` 约束 hard-pair 和因子关系；
- `lambda_h`、`lambda_r` 是损失权重。

大白话理解：我们希望模型重点修正弱类，但不能因为修正弱类就破坏原来已经分类正确的类别。

## 12. 输出与指标

模型输出 27 个 logits：

```text
z = [z_1, z_2, ..., z_27]
```

再经过 softmax 得到概率：

```text
p_c = exp(z_c) / sum_j exp(z_j)
```

预测类别为：

```text
y_hat = argmax_c p_c
```

当前完整测试集结果：

```text
Top-1      = 90.632%
Macro-F1  = 88.920%
Weighted-F1 = 90.654%
Weakest class = water_concrete_slight
Weakest-class F1 = 75.693%
```

其中：

- Top-1 表示总体分类准确率；
- Macro-F1 先算每个类别的 F1，再对 27 类平均，更能反映弱类表现；
- weakest-class F1 表示 27 个类别中表现最差的类别 F1。

## 13. 当前算法的主要创新点

当前算法相对于普通分类网络的主要不同点是：

1. 把 RSCD 27 类显式拆成状态、材质、粗糙度三个因素。
2. 用 PhysicsTexture 提取湿滑、反光、粗糙、纹理擦除等可解释物理线索。
3. 用 LocalPhysicsField 保留局部物理证据，而不是只看全局平均。
4. 用 SemanticPhysicsAttention 让不同类别关注不同物理证据。
5. 用耦合因子建模表达二阶和三阶组合关系。
6. 用 hard-pair gated calibration 只修正高风险混淆边界。
7. 用 anchor/no-flip 保护机制避免修正弱类时伤害强类。

## 14. 当前局限

当前模型的 Top-1 仍低于部分公开 SOTA，主要瓶颈不是大类识别，而是细粒度组合边界：

- `water` 和 `wet` 的视觉差异很小；
- 水膜可能遮挡 concrete 的纹理；
- slight 粗糙度处于 smooth 和 severe 之间；
- 某些局部证据只占 patch 的一小部分。

因此，后续最有希望的方向不是继续堆叠晚期 head，而是改进早期或中期特征提取机制，让主干网络在纹理、反光、粗糙度、材质耦合处更早地区分有效证据。
