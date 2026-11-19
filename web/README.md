# Chun-Li One Credit 网页

从仓库根目录运行 `python -m http.server 8765 -d web`，打开 [本地网页](http://127.0.0.1:8765/index.html)。详细用法见[项目 README](https://github.com/Datou/ChunLi-OneCredit)和[存档制作](../docs/savestates.md)。

选择 ROM 后自动加载该关开局并开打。起始关卡选择器可以重开指定对手；策略动作栏显示前、重腿和近身投等按键语义。电脑与手机共用「按住切断输入」按钮。按住时丢弃排队指令，春丽接收中立输入。松手后恢复策略。默认每个物理帧有 5% 的概率增加 1–5 帧随机输入延迟。

网页按原生模拟器帧率推进，每 4 帧执行一次策略前向。策略优先使用 WASM SIMD，保留纯 JavaScript 路径。44.1 kHz 游戏音频由 AudioWorklet 独立播放；浏览器要求用户手势时会显示「启用声音」按钮。
