// Audio callback logic tests, not browser automation or microphone access.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const code = fs.readFileSync(__dirname + '/static/capture-worklet.js', 'utf8');
const results = [];
for (const rate of [16000, 44100, 48000, 96000]) {
  for (const hop of [20, 200, 300]) {
    const messages = [];
    const context = {sampleRate: rate, currentFrame: 0, Float32Array,
      AudioWorkletProcessor: class {constructor() {this.port = {postMessage: d => messages.push(structuredClone(d))};}},
      registerProcessor: (name, cls) => context.Processor = cls};
    vm.createContext(context); vm.runInContext(code, context);
    const processor = new context.Processor();
    const frameCount = Math.round(rate * hop / 1000), blocks = hop === 200 ? 150 : 7;
    const limit = frameCount * blocks, startAt = 256 / rate;
    processor.port.onmessage({data:{action:'active',value:true,hopMs:hop,startAt,limitFrames:limit}});
    for (let begin = 0; begin < limit + 512; begin += 128) {
      context.currentFrame = begin;
      const output = [new Float32Array(128).fill(1)];
      processor.process([[new Float32Array(128).fill(.1),new Float32Array(128).fill(.3)]], [output]);
      assert(output[0].every(x => x === 0), 'Capture must not play unfiltered input');
    }
    const packets = messages.filter(x => x.samples), done = messages.filter(x => x.done);
    assert.strictEqual(packets.length, blocks); assert.strictEqual(done.length, 1);
    assert.strictEqual(done[0].blocks, blocks);
    for (const [i,p] of packets.entries()) {
      assert.strictEqual(p.sequence,i); assert.strictEqual(p.samples.length,frameCount);
      assert(Math.abs(p.startTime - (256 + i * frameCount) / rate) < 1e-10);
      assert(p.samples.every(x => Math.abs(x - .2) < 1e-6), 'Downmix/sample continuity');
    }
    assert.strictEqual(processor.captured,limit); assert.strictEqual(processor.active,false);
    const oldCount = messages.length;
    processor.port.onmessage({data:{action:'active',value:false}});
    processor.process([[new Float32Array(128).fill(.9)]], [[new Float32Array(128)]]);
    assert.strictEqual(messages.length,oldCount,'No capture after Stop');
    results.push({sample_rate_hz:rate,hop_ms:hop,blocks,frames:limit,
      exact_30s_calibration:hop === 200, stop_silences_capture:true, capture_graph_silent:true});
  }
}
fs.writeFileSync(__dirname+'/worklet_verification.json',JSON.stringify({scope:'AudioWorklet logic; no physical microphone',checks:results},null,2)+'\n');
console.log('AudioWorklet sample counts, mono continuity, finite capture and Stop passed for 12 configurations.');
