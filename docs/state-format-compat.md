# 网页存档兼容性

训练与当前网页都使用 [Gym Retro v0.8.0 的 Genesis Plus GX 1.7.5](https://github.com/openai/retro/tree/v0.8.0/cores/genesis)。x64 与 wasm32 的存档缓冲区都是 1,036,288 字节，音频芯片和 Z80 结构中序列化的指针及填充宽度不同。WASM 读取前需[重打包](../scripts/web_pack_retro_states.mjs)。

打包保留工作内存、显存、VDP 运行状态及 68K 上下文。FM 声道和 Z80 的非指针字段按原值重排；音频连接和 Z80 中断回调由 core 自己重建。文件头仍为 `GENPLUS-GX 1.7.5`，卡带映射从 x64 偏移 144,613 移至 wasm32 偏移 144,361。

用于历史对照的 1.7.6 核心和开局仍在 `web/cores/` 和 `web/cores/fixtures/ladder/`。这些开局用 1.7.6 模板替换过 CPU 控制状态，固定输入回放可以出现轨迹分叉。相关实验见 [Ken 胜率复测](ken-winrate.md)。

重新构建当前网页核心时，将 Gym Retro v0.8.0 的 `cores/genesis` 作为 `GxRoot`。

```powershell
./scripts/build_gx_wasm.ps1 -GymRetro -GxRoot <gym-retro>/cores/genesis -OutDir web/cores/gym-retro
```
