class GenesisAudio extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blocks = [];
    this.offset = 0;
    this.buffered = 0;
    this.bufferTarget = Math.round(sampleRate * 0.09);
    this.started = false;
    this.played = 0;
    this.underruns = 0;
    this.nextReport = 0;
    this.port.onmessage = ({ data }) => {
      this.blocks.push(data);
      this.buffered += data.length / 2;
      while (this.buffered > sampleRate * 0.25 && this.blocks.length > 1) {
        this.buffered -= (this.blocks.shift().length - this.offset) / 2;
        this.offset = 0;
      }
    };
  }

  process(_inputs, [output]) {
    if (!output.length) return true;
    const [left, right] = output;
    if (!this.started && this.buffered >= this.bufferTarget) this.started = true;
    if (this.started) {
      let written = 0;
      while (written < left.length && this.blocks.length) {
        const head = this.blocks[0];
        const available = Math.min(left.length - written, (head.length - this.offset) / 2);
        for (let i = 0; i < available; i++) {
          left[written] = head[this.offset++] / 32768;
          right[written++] = head[this.offset++] / 32768;
        }
        this.buffered -= available;
        if (this.offset === head.length) {
          this.blocks.shift();
          this.offset = 0;
        }
      }
      this.played += written;
      if (written < left.length) {
        this.underruns++;
        this.started = false;
      }
    }
    if (currentFrame >= this.nextReport) {
      this.port.postMessage({ framesOut: this.played, underruns: this.underruns });
      this.nextReport = currentFrame + sampleRate;
    }
    return true;
  }
}

registerProcessor("genesis-audio", GenesisAudio);
