class TSECapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = Math.round(sampleRate / 5);
    this.buffer = new Float32Array(this.size);
    this.offset = 0;
    this.active = false;
    this.sequence = 0;
    this.port.onmessage = ({data}) => {
      if (data.action === 'active') {
        this.active = data.value;
        this.size = Math.round(sampleRate * (data.hopMs || 200) / 1000);
        this.buffer = new Float32Array(this.size);
        this.offset = 0;
        this.sequence = 0;
        this.startAtFrame = Math.round((data.startAt || 0) * sampleRate);
        this.limit = data.limitFrames || Infinity;
        this.captured = 0;
      }
    };
  }
  process(inputs, outputs) {
    // The capture graph is silent; only processed audio is connected to playback.
    for (const channel of outputs[0] || []) channel.fill(0);
    const channels = inputs[0];
    if (!this.active || !channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      if (currentFrame + i < this.startAtFrame || !this.active) continue;
      if (this.offset === 0) this.startFrame = currentFrame + i;
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.buffer[this.offset++] = value;
      this.captured++;
      if (this.offset === this.size) {
        const samples = this.buffer;
        this.port.postMessage({sequence: this.sequence++, startTime: this.startFrame / sampleRate, samples}, [samples.buffer]);
        this.buffer = new Float32Array(this.size);
        this.offset = 0;
      }
      if (this.captured === this.limit) {
        this.active = false;
        this.port.postMessage({done: true, blocks: this.sequence});
      }
    }
    return true;
  }
}
registerProcessor('tse-capture', TSECapture);
