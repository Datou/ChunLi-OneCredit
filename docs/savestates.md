# 存档制作

训练与网页使用 Gym Retro 的 Genesis Plus GX 1.7.5。每份训练存档是 gzip 压缩的原始 core 状态，解压后为 1,036,288 字节，开头为 `GENPLUS-GX 1.7.5`。网页文件需要把 x64 存档重新打包为 WASM 的布局。

## 手动保存第 1 关

1. 在 [Gym Retro Integrator](https://retro.readthedocs.io/en/latest/integration.html) 或能调用同一版 Genesis Plus GX core 的前端载入 USA 版 ROM。选择 Champion 模式和春丽。在镜像战的第 1 局 `ROUND 1` 开场动画期间暂停，趁角色还无法动作时保存。
2. 导出原始 core 状态。Integrator 已导出的 gzip `.state` 可直接使用。如果模拟器保存了自己的容器格式，请用该前端的原始 core 状态导出功能，或在同版 core 的 Integrator 里重存。
3. 将存档转换为训练文件，然后逐帧推进到开场锁解除的瞬间。

```bash
python src/convert_state_to_retro.py --input mirror.state --output generated_states/Stage01.ChunLi.state
python -m chunppo.make_states --stages 1
```

转换器验证解压后的长度与版本，再封装 gzip。其他 core 的状态布局不同，需要在兼容的 core 中重新保存。

## 生成第 2–12 关

**自动生成。** 从第 1 关可行动开局运行公开权重，按原游戏推进和经过奖励关。后续每关按帧检查，遇到满血、0:0、开场解锁且向前输入能生效时立即保存。采集器在另一个 Gym Retro 进程中验证候选存档，赛程中的模拟器继续自然推进。

```bash
python -m chunppo.capture_ladder --model chunli-ppo.zip --offset 87 --jitter 0.05
```

该命令在本项目使用的春丽镜像起点上采齐了 12 关。不同的手动起点会改变赛程。若某次赛程输掉整场，已验证的开局照常保存，可以换 `--offset` 再跑，或手动补齐。

**手动生成。** 在模拟器中以春丽打赢比赛，跟随游戏自然进入下一关。每关在第 1 局开场锁还有效时暂停并保存，逐一用上面的转换器写入 `generated_states/`。全部齐备后运行 `python -m chunppo.make_states`，一次性逐帧转换 12 份开局。也可用 `--stages 3 8` 只重存指定关卡。

训练读取 `generated_states/controllable/` 中的文件。重新载入验证满血 176:176、局分 0:0、对手身份和前进输入。验证摘要写在同目录的 `manifest.json`。关卡按对手编号，文件依次是：

| 关卡 | 文件 |
| --- | --- |
| 01–04 | `Stage01.ChunLi.state`、`Stage02.Zangief.state`、`Stage03.Dhalsim.state`、`Stage04.Ryu.state` |
| 05–08 | `Stage05.EHonda.state`、`Stage06.Blanka.state`、`Stage07.Guile.state`、`Stage08.Ken.state` |
| 09–12 | `Stage09.Balrog.state`、`Stage10.Vega.state`、`Stage11.Sagat.state`、`Stage12.Bison.state` |

## 生成网页存档

网页已经附带 12 份开局。用 Node.js 将生成好的可行动训练存档打包到网页目录。

```bash
node scripts/web_pack_retro_states.mjs generated_states/controllable web/cores/fixtures/gym-retro
python -m http.server 8765 -d web
```

脚本会检查原始存档的版本、布局、双方血量和输入锁，然后按 WASM 的字段布局重打包。生成的 `web/cores/fixtures/gym-retro/stage01.state.gz` 至 `stage12.state.gz` 保留同一个开局帧。详情见[网页存档兼容性](state-format-compat.md)。
