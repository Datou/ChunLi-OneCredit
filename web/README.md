# Chun-Li One Credit 网页

从仓库根目录运行 `python -m http.server 8765 -d web`，打开 [本地网页](http://127.0.0.1:8765/index.html)。详细用法见[项目 README](https://github.com/Datou/ChunLi-OneCredit)和[存档制作](../docs/savestates.md)。

选择 ROM 后自动开打。起始关卡选择器可以重开指定对手；策略动作栏显示前、重腿和近身投等按键语义。在手机上长按画面下的延迟按钮，或在桌面端按住空格，为新输入增加 6 帧延迟；松开后已排队输入仍按顺序释放。默认每个物理帧有 5% 的概率增加 1–5 帧随机输入延迟。

网页按原生模拟器帧率推进，每 4 帧执行一次策略前向。策略优先使用 WASM SIMD，保留纯 JavaScript 路径。44.1 kHz 游戏音频由 AudioWorklet 独立播放；浏览器要求用户手势时会显示「启用声音」按钮。
