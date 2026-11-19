# 训练与评测

最终策略在约 **2.70 亿个决策步**后冻结。一次决策执行 4 个模拟器帧；回合间过场不计为策略决策步。

## 复用同一个策略覆盖全部 12 关

1. **观测和动作。** NatureCNN 接收从 `t−16`、`t−8`、`t` 取样的 84×84 运动前景。独立分支编码最近 24 个发出的动作，帮助判断蹲蓄力和连打。策略每 4 帧决策一次，选择 27 个春丽动作之一。普通攻击按 3 帧后松开 1 帧，蓄力释放先给方向再给攻击键。控制层从模拟器读取横坐标，将相对的前后映射成左右。
2. **基础 PPO。** 14 个采样环境轮换覆盖 12 关。开局插入 0–30 帧随机走位，训练中保留对战快照供重置抽样。奖励来自双方血量变化、KO 和步长代价。每个环境独立收集 512 个决策，再统一更新；图像 rollout buffer 使用 uint8。基础训练预算为约 2 亿步。
3. **自然赛程微调。** 用冻结策略实际打一币赛程，保存转场后开局和交战中的完整快照。快照同时还原模拟器、观测与动作历史、记分状态和延迟队列。追加约 7,000 万步 PPO，最终阶段提高古烈、肯、沙加特和将军的采样权重，并从肯的实际对局中补充受伤和恢复局面。微调混用 0%、1%、3%、5%、8% 的输入抖动。发生抖动时，该物理帧的输入延后 1–5 帧，队列禁止后来的输入超车。

训练脚本为 [train.py](../chunppo/train.py)、[collect_campaign_bank.py](../chunppo/collect_campaign_bank.py) 和 [finetune.py](../chunppo/finetune.py)。以下命令使用本地生成的存档和快照库，展示有效流程。训练快照由各阶段的权重运行重新采集。

```bash
python -m chunppo.train --out runs/base --total 200000000 --checkpoint 2500000
python -m chunppo.collect_campaign_bank --model runs/base/final.zip --out runs/natural-bank --seed 6051105
python -m chunppo.finetune --resume runs/base/final.zip --bank runs/natural-bank --out runs/natural --start-timesteps 200001536 --total 220001536
```

后续继续使用相同的采集和微调流程，补充自然赛程中难关的中盘快照。后续阶段的 `--stage-copies 0,0,0,0,0,0,2,6,0,0,3,1` 为古烈、肯、沙加特和将军增加采样。最终训练使用学习率 0.00005、熵系数 0.005、PPO clip 0.1。原始训练快照与中间权重独立于公开终版权重，因此从新建存档训练应以独立评测结果为准。

## 分开测量单场与连续通关

**连续赛程。** 从第 1 关加载一次存档；每场三局两胜，输了整场便结束。赢了整场由游戏自然推进。锁步评测使用确定性 argmax、31–90 帧留出开局偏移、0 帧固定延迟，分别测试 0% 和 5% 的随机抖动。转场后的首个决策使用当前画面，与训练时序对齐。

```bash
python -m chunppo.evaluate_one_credit --model chunli-ppo.zip --out runs/one-credit --campaigns 100 --seed 9071106 --workers 4 --sync-round-observation
```

**单场评测。** 每个对手独立读档，分别记录该场胜负。该指标衡量单个对手开局，完整通关率由连续赛程测量。

**实时实战。** 原生录像使用接近 59.923 Hz 的模拟器时钟，游戏独立于策略前向推进，迟到的输入包沿用上一输入。一次选定条件下的记录在 5% 抖动下自然赢满 12 场。浏览器演示采用 4 帧一决策的网页调度；README 中的统计口径是 Python 锁步评测。
