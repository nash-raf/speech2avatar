class WSAudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.startBufferFrames = Math.max(128, opts.startBufferFrames || 2400);
    this.queue = [];
    this.queueOffset = 0;
    this.bufferedFrames = 0;
    this.playedFrames = 0;
    this.underflows = 0;
    this.started = false;
    this.reportCountdown = 0;

    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "push" && msg.pcm) {
        const pcm = msg.pcm instanceof Float32Array ? msg.pcm : new Float32Array(msg.pcm);
        if (pcm.length) {
          this.queue.push(pcm);
          this.bufferedFrames += pcm.length;
        }
      } else if (msg.type === "reset") {
        this.queue = [];
        this.queueOffset = 0;
        this.bufferedFrames = 0;
        this.playedFrames = 0;
        this.underflows = 0;
        this.started = false;
      }
    };
  }

  _report(force = false) {
    this.reportCountdown += 1;
    if (!force && this.reportCountdown < 8) return;
    this.reportCountdown = 0;
    this.port.postMessage({
      type: "stats",
      playedFrames: this.playedFrames,
      bufferedFrames: this.bufferedFrames,
      underflows: this.underflows,
      started: this.started,
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output.length) {
      return true;
    }

    const channel = output[0];
    channel.fill(0);

    if (!this.started && this.bufferedFrames >= this.startBufferFrames) {
      this.started = true;
      this._report(true);
    }

    if (!this.started) {
      this._report(false);
      return true;
    }

    let written = 0;
    while (written < channel.length && this.queue.length) {
      const current = this.queue[0];
      const available = current.length - this.queueOffset;
      const needed = channel.length - written;
      const take = Math.min(available, needed);
      channel.set(current.subarray(this.queueOffset, this.queueOffset + take), written);
      written += take;
      this.queueOffset += take;
      this.bufferedFrames -= take;
      this.playedFrames += take;

      if (this.queueOffset >= current.length) {
        this.queue.shift();
        this.queueOffset = 0;
      }
    }

    if (written < channel.length) {
      this.underflows += 1;
      this._report(true);
    } else {
      this._report(false);
    }

    return true;
  }
}

registerProcessor("ws-audio-playback", WSAudioPlaybackProcessor);
