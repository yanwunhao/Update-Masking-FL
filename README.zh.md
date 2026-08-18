# fed_diversity — 中文说明文档

FedAvg 伪梯度的**每轮梯度多样性**测量工具。这是论文里唯一的新实验结果,
**被测量的量的正确性优先于其他一切**。

```
Δ_S = Σ_{k∈C} ‖Δ_k‖² / ‖Σ_{k∈C} Δ_k‖²        Δ_k = w_broadcast − w_k_final
```

其中 C 是本轮被采样的客户端队列(cohort),|C| = K_c。上报 `1/Δ_S`,取值范围 `[1, K_c]`:

- `1/Δ_S = K_c` —— 所有客户端返回完全相同的伪梯度(完全一致)
- `1/Δ_S = 1` —— 各客户端伪梯度两两正交(完全不相关)

每轮同时记录两个变体:

| 列名 | 含义 |
|---|---|
| `delta_s_unweighted` | 上式原样,**对应理论** |
| `delta_s_weighted` | 每个 `Δ_k` 先乘以 `n_k / mean_j(n_j)`,**对应 FedAvg 实际聚合的东西** |

两者在客户端样本数不均时(即 α 小时)会分开。`delta_s_*` 是比值本身,
`inv_delta_s_*` 是它的倒数,两个方向都写进文件,读的人不用猜。

---

## 一、必读:数字怎么解释

### test 2 才是正确性对照

给每个客户端**相同的数据分片 + 相同的 RNG 流**,E=1 → 每个 `Δ_k` 必然相同 →
`1/Δ_S` **精确等于** `K_c`。这一个测试同时抓住三类 bug:参数展平顺序、
buffer 混入、cohort 选择。**它失败就停下,别的都没有意义。**

### α=100 是噪声地板测量,不是正确性检查

它**不会**接近 `K_c`。本代码实测:

| 配置 | `1/Δ_S` |
|---|---|
| α=100,10 轮,K_c=10 | ≈ **3.25** |
| α=100,200 轮,K_c=10(最后 50 轮) | ≈ **1.19** |
| α=100,batch=full,E=1(最小噪声角) | ≈ **9.67**(前几轮) |

分解结果解释了原因:`‖μ̂‖²` 从第 25 轮起基本不变(0.0047 → 0.0040),
而 `σ̂²` 单调增长(0.052 → 0.186),同期精度升到 68%。
**把比值压到地板的是局部更新噪声,不是数据异质性。**

任何期待 "α=100 ≈ K_c" 的人都会以为代码坏了 —— 它没坏。
这也正是 `local-steps` 成为主 preset、α 被降级为鲁棒性检查的原因。

### 分解量(不额外增加计算)

令 `A = (1/K_c)Σ‖Δ_k‖²`,`B = ‖Δ̄‖²`,则 `1/Δ_S = K_c·B/A`。
设 `Δ_k = μ + ε_k`(共享方向 + 每客户端独立噪声):

```
sigma2_hat = (A − B) / (1 − 1/K_c)
mu2_hat    = A − sigma2_hat
r_hat      = mu2_hat / sigma2_hat
one_over_delta_s_predicted = (K_c·r + 1)/(r + 1)
```

**注意 `one_over_delta_s_predicted` 是代数恒等式**,不是对模型的独立检验 ——
把 r 代进去必然塌回 `K_c·B/A`。之所以每轮断言它,是因为不相等就说明
`decompose()` 和 `_ratio()` 的算术走岔了,那是 bug。

估计量假设 `ε_k` **跨客户端独立且等方差**,两条都没被验证、也都只是近似:
共享类别的客户端噪声相关,`n_k` 不同的客户端方差不同。请把 `mu2/sigma2`
当作 cohort 离散度的两参数摘要,而不是对某个潜在真值的测量。

当客户端接近相同时 `sigma2_hat` 会 ≤ 0(此时 B ≥ A),这时 `r_hat` 用
`SIGMA2_EPS` 兜底,并置 `sigma2_clamped = true` 标志 —— **绝不隐藏**。

---

## 二、文件说明

### 核心

| 文件 | 行数 | 说明 |
|---|---|---|
| **`fed_diversity.py`** | 990 | 主程序。跑**一个** `(α, seed, batch, E)` 配置、**一个**设备。包含模型、数据、Dirichlet 划分、FedAvg 循环、测量与分解、JSONL 输出。没有 DDP / DataParallel —— 并行来自跑多个独立进程。 |
| **`run_sweep.py`** | 523 | 调度器。把 preset 网格分发到多个设备上,每设备最多 `--per-gpu` 个并发进程,退出即补位。维护 `manifest.json`,支持 `--dry-run` / `--status` / 断点续跑 / 结束汇总。 |
| **`test_fed_diversity.py`** | 404 | 13 个测试,四类正确性闸门(见第三节)。 |
| **`check.py`** | 79 | `make check` 的实现:数据校验 + 全部测试 + 端到端 smoke,输出单行 PASS/FAIL,任何失败非零退出。**新机器上第一个要跑的东西。** |

`fed_diversity.py` 内部几个关键函数:

- `param_names()` / `get_flat()` / `set_flat()` —— 固定且**每次调用都重新断言**的参数顺序;
  同时断言模型 **buffer 数为 0**(直接防止 BatchNorm 溜回来)。
- `_ratio()` —— float64 累加的比值计算,含分母下溢保护。
- `decompose()` / `assert_decomposition()` —— 上面那套分解。
- `diversity()` —— 组装 unweighted / weighted 两个变体的全部列。
- `aggregate()` —— **服务器聚合单独成一个函数**,将来插 mask mode 不用动训练循环。
- `keyed_generator()` —— 由整数 key 派生的 **CPU** 生成器(见第五节)。
- `resolve_batch_size()` —— `"full"` 解析为整个分片大小。

### 数据与合并

| 文件 | 行数 | 说明 |
|---|---|---|
| **`prepare_data.py`** | 126 | 单进程下载并校验 CIFAR-10,打印需要拷贝到另一台机器的文件。`--check` 只校验不下载。 |
| **`merge_results.py`** | 200 | 合并多台机器的结果目录。检测**配置冲突**(同一 α/seed/batch/E 在两台机器上都跑过)并**报告差异**而不是静默挑一个;打印所有不同的 torch/CUDA/dtype/compile 环境。 |

**为什么 `prepare_data.py` 必须单独跑**:torchvision 下载不加锁。在还没有数据的
机器上直接启动 sweep,N 个进程会同时写同一个压缩包、互相交错,得到一个截断的 tar,
之后每个 run 都在数据加载器里挂掉 —— 症状看起来像数据 bug,能浪费一下午。
`run_sweep.py` 会**拒绝**在数据缺失时启动,并打印确切的补救命令。

### 绘图

| 文件 | 行数 | 说明 |
|---|---|---|
| **`plot_primary.py`** | 122 | 主图 `fig_local_steps.png`,两个共享横轴的面板。 |
| **`plot_alpha.py`** | 73 | 次图 `fig_alpha.png`,`1/Δ_S` vs 轮次,每个 α 一条线。 |
| **`plot_common.py`** | 131 | 共享的读取、配色、坐标轴样式。 |

主图两个面板:

- **(a)** `1/Δ_S` vs 轮次,每个 `(batch, E)` 一条线,跨 seed 中位数 + min-max 带,
  `y=1` 和 `y=K_c` 两条参考线,对数纵轴。
- **(b)** `‖μ̂‖²` 和 `σ̂²` vs 轮次(代表性配置),对数纵轴。
  **解释在这个面板上** —— (a) 的比值只是 `r` 的函数,它无法显示是**哪一项**在动;
  实测下降来自 `σ̂²` 增长而非 `μ̂` 崩塌,只有 (b) 能区分这两者。

配色用经过校验的参考调色板,分类槽位**固定顺序、绝不循环、绝不按排名重分配**。
若干槽位在浅色背景上对比度低于 3:1,因此全程遵守 relief 规则:
每条线在右端直接标注,身份**不单靠颜色**承载。

### 打包

| 文件 | 说明 |
|---|---|
| **`requirements.txt`** | 钉死版本:torch 2.8.0 / torchvision 0.23.0 / numpy 2.1.2 / matplotlib 3.10.9 |
| **`Makefile`** | `make data` / `check` / `dry-run` / `status` / `primary` / `alpha` |
| **`RUNBOOK.md`** | 英文操作手册 |
| **`README.zh.md`** | 本文件 |

---

## 三、测试(任何正式 run 之前必须通过)

| # | 测试 | 性质 |
|---|---|---|
| 1 | **合成、精确**。K 个相同向量 → `Δ_S = 1/K`;K 个正交等模向量 → `Δ_S = 1`。断言到 **1e-12**。 | 正确性 |
| 2 | **端到端、精确**。全部客户端同分片同 RNG,E=1 → `1/Δ_S = K_c`,到 ~1e-5。 | **正确性对照** |
| 3 | **集成、近似**。α=100 真实 CIFAR-10,只报数不做紧界断言。 | **噪声地板,非正确性检查** |
| 4 | **跨设备确定性**。cpu 与 cuda 在前几轮的 `Δ_S` 上一致。 | 可移植性 |

另有:分解恒等式与 σ² 回收、`full` 批解析、下溢标志、边界断言双向触发、
展平往返与零 buffer、Dirichlet 划分覆盖性与最小样本数。

**test 2 失败就停下并修复。不要在部分通过的情况下继续。不要为了让测试变绿而放宽容差。**

---

## 四、操作顺序

### 0. 安装

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### 1. 数据(每台机器一次,先于一切)

```bash
python prepare_data.py --data-root ./data
```

链路好时约 2 分钟;上游经常限速到 ~250 KB/s,预留 15 分钟。
两台机器**没有共享文件系统**,所以两边都要跑,或者拷贝它打印的 `cifar-10-batches-py/`。

### 2. `make check` —— 闸门

```bash
make check                 # CPU
make check DEVICE=cuda:0   # GPU
```

CPU 约 4 分钟,GPU 约 1 分钟。跑数据校验 + 全部测试 + 端到端 smoke。
新机器上第一件事,以及每次改动之后。

本机实测:

```
CPU:     PASS -- 3/3 stage(s) in 124s on cpu      (13/13 tests)
cuda:0:  PASS -- 3/3 stage(s) in  47s on cuda:0   (13/13 tests)
```

### 3. Smoke(一分钟内跑通全路径)

```bash
python fed_diversity.py --alpha 1.0 --seed 0 --smoke --device cpu
```

5 轮 / 20 客户端 / 采样 3 个 / 每客户端 200 样本。笔记本 CPU 友好。**不是测量结果。**

### 4. 先看计划

```bash
python run_sweep.py --preset local-steps --dry-run
```

列出每个配置、预计耗时、哪些已完成。

### 5. 跑

```bash
tmux new -s sweep
python run_sweep.py --preset local-steps --devices cuda:0 --per-gpu 3
python run_sweep.py --preset alpha-robustness --devices cuda:0 --per-gpu 3
```

**务必用 tmux 或 screen。** 这些 run 要跑几个小时,SSH 掉线会杀掉整个进程树,
所有正在跑的配置全部丢失。

**选 `--devices` 和 `--per-gpu` 之前先看 `nvidia-smi`。** 这是共享机器。
每个进程约需 1.2–1.5 GB 加激活值;`--per-gpu 3` 适合 8 GB 的 4060。
在 48 GB 的 A6000 上可以调高 —— 但要**根据 sweep 打印的实测峰值显存来调,不要猜**。

另开一个 shell 看进度:

```bash
python run_sweep.py --status
```

**续跑是自动的,且只在配置级别**:JSONL 以完成记录结尾就跳过。
**没有 run 内 checkpoint** —— 被杀最多损失一个配置。
RNG 状态序列化只会增加 bug 面积,在单个 run 不到一小时的情况下没有收益。

**没有锁文件**:两台机器不共享文件系统,不可能争抢同一个配置。
分工靠在各自机器上跑不同 preset 或 `--seeds`,事后用 `merge_results.py` 合并。

### 6. 跨机器合并

```bash
python merge_results.py results_4060 results_a6000 --out merged
```

有冲突时会报告实测差异并**拒绝合并**,直到你用 `--on-collision first|longest` 明确选择。
同时打印所有不同环境 —— `tf32` / `amp` / `--compile` 会把 `Δ_S` 扰动约 7e-4 相对量,
**不能**和默认模式的结果混在一起。

### 7. 出图

```bash
python plot_primary.py --results-dir merged     # fig_local_steps.png
python plot_alpha.py   --results-dir merged     # fig_alpha.png
```

---

## 五、Preset 与参数

### Preset

| preset | 网格 | run 数 |
|---|---|---|
| `local-steps` **【主】** | α=100,200 轮,seeds {0,1,2},batch {32,128,full} × E {1,2,5} | **27** |
| `alpha-robustness` 【次】 | α {100,1,0.1},200 轮,seeds {0,1,2},batch 32,E=2 | 9 |
| `smoke` | α {100,0.1},5 轮,seed 0 | 2 |

> **注意**:规格文档写的是 45 个 run,但 3 batch × 3 E × 3 seeds = **27**。
> 这里按字面写的轴和 seed 数实现为 27,没有擅自把 seed 改成 5 个来凑 45。

`local-steps` 沿一个**方向已知**的轴改变 `r`:局部步数越少、批越大,累积的局部噪声越少,
`r` 越大,`1/Δ_S` 越高。`full` + E=1 是最小噪声角,应当最高。
**如果那里没有明显上升,就说明分解模型是错的** —— 这件事值得在别的都跑之前先知道。

### 必须暴露的机器相关参数(都有默认值)

```
--device            "cpu" | "cuda" | "cuda:N"
--devices           run_sweep.py 用,逗号分隔
--per-gpu           每设备并发进程数(默认 3,按 8 GB 定)
--dtype             "fp32" | "tf32" | "amp"        默认 fp32
--compile           torch.compile 开关                默认关
--data-device       "device" | "cpu"                默认 "device"
--num-threads       torch.set_num_threads
--data-root, --out-dir
```

### 绝不能随机器变化的实验参数

学习率、客户端数、cohort 大小、轮数、α、seed、**局部批大小**、局部 epoch 数。

局部批大小是**实验轴**(见 preset),但**绝不会为了塞进显存而自动缩小** ——
放不下就带着明确信息报错,指向 `--per-gpu` 和 `--data-device cpu`,
而不是悄悄改小然后产出一个无法与其他 run 比较的数字。

### 确定性不依赖硬件

所有随机抽取都在 **CPU 生成器**上完成再搬到设备,key 为 `(seed, round)` 或
`(seed, round, client)`。**CUDA 生成器用种子 s 产生的流与 CPU 生成器用同一个 s 不同**,
所以在设备侧抽 batch 顺序或裁剪偏移会让 run 变成机器特定的。
轮循环里不消费任何全局 RNG 状态 —— 全局状态的消费顺序会随设备、线程数、库版本漂移。

实测:cohort 与分片大小跨 CPU/CUDA **完全相等**;`Δ_S` 第 0 轮一致到 **1.6e-6**,
第 2 轮漂到 ~5.2e-3。这不是 RNG bug:同一批数据在完整 float64 下,
CPU 与 CUDA 的一次前向+反向一致到 **2.1e-15**。是 fp32 卷积核末位不同,
被 SGD 的混沌性每轮放大约 30 倍。

---

## 六、已知行为(不要被吓到)

### `1/Δ_S ≥ 1` 不是定理

只有 `≤ K_c` 是(Cauchy–Schwarz)。下界成立当且仅当交叉项平均非负,
即 `Σ_{j≠k}⟨Δ_j, Δ_k⟩ ≥ 0`。强异质 cohort 完全可能产生净相消的伪梯度,
那时 `1/Δ_S < 1` 是**真实测量**,不是错误。

按规格要求,边界断言在越界时**硬报错**:不 clamp,不警告后继续。
出问题的那一行**先落盘再抛异常**,所以证据不会丢。

α=100 实测最低到 **1.0778**,距下界只剩 **0.078** 裕度 ——
**低 α 的 run 触发这个断言是很现实的可能。** 真发生时先读那一行再判断是不是 bug;
`fed_diversity.py` 里的 `LOWER_BOUND` 是那个需要**刻意**去动的旋钮。

### 其他

- **`sigma2_clamped`** 标记客户端接近相同、`σ̂²` 算出非正的轮次。
- **`Infinity`** 出现在分母下溢行的 `delta_s_*` 里。`json.loads` 能正确解析,
  该行同时带 `denominator_underflow_*: true`。
- **全批配置很慢且吃显存**:实测 `full`/E=1 是 **37.0 s/轮**、峰值 **7.5 GiB**,
  而 batch 32/E=2 是 10.5 s/轮。全批**步数更少、处理样本更少**,却慢 3.5 倍 ——
  ~500 张图的 ResNet-18 单步是访存受限的,逼近显存上限时更差。
  **在 4060 上跑全批配置必须 `--per-gpu 1`。**
  `--dry-run` 的估计基于两个实测锚点,只能当量级参考。

---

## 七、输出格式

每个 run 一个 JSONL,每轮一条记录:

```
round, alpha, seed, cohort, K_c, n_k,
delta_s_unweighted, delta_s_weighted,
inv_delta_s_unweighted, inv_delta_s_weighted,
mu2_hat, sigma2_hat, r_hat, one_over_delta_s_predicted, sigma2_clamped,
norm_sq_of_sum_*, sum_of_sq_norms_*,
delta_norm_mean_*, delta_norm_std_*,
denominator_underflow_*,
test_acc(每 10 轮), round_seconds, elapsed_seconds
```

头记录含完整 config 与环境:`hostname`、`device_name`、`torch_version`、
`cuda_version`、`cudnn_version`、`dtype`、`compile`、`data_device`、`num_threads`、
`per_gpu`、`git_hash`、`python`,以及参数名列表和各客户端样本数。
尾记录含 `total_seconds`、`seconds_per_round`、`peak_memory_mb`、`final_test_acc`。

---

## 八、不做的事

不做 mask mode、不做优化器变体、不做第二个数据集、
不做周期性评估之外的精度扫描、不做 SLURM 模板、不做分布式训练。

DDP / DataParallel 对这个负载是错的 —— 并行性来自互相独立的进程。
聚合步骤已经单独成一个函数,将来插 mask mode 不需要重构循环。
