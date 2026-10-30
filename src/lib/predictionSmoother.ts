// Temporal smoothing for SIBI letter prediction.
//
// Turns noisy per-frame predictions into one letter per held sign.
//
// Design goals (learned from real use):
//  - A new sign must take effect FAST: the window is only the last few frames,
//    so the previous letter stops being "majority" almost immediately.
//  - The hold time is adaptive: a confident, clean sign locks in ~350 ms; a
//    shaky one gets more time. There is no fixed 2 second requirement.
//  - Natural hesitations (speech pauses, tiny hand adjustments) do not wipe
//    progress. Only a long gap (>700 ms without any prediction) restarts it.
//  - Ambiguous flicker (two letters neck and neck) never commits.

export interface SmootherOptions {
	windowSize?: number;
	minFrames?: number;
	minConfidence?: number;
	cooldownMs?: number;
	holdMs?: number;
}

export class PredictionSmoother {
	private window: { label: string; conf: number; t: number }[] = [];
	private lastCommitAt = 0;
	private lastPushAt = 0;
	private currentLabel = '';
	private labelStartAt = 0;
	private opts: SmootherOptions;
	lastEmitted: { label: string; conf: number } | null = null;
	progress = 0;

	constructor(opts: SmootherOptions = {}) {
		this.opts = opts;
	}

	reset() {
		this.window = [];
		this.progress = 0;
		this.lastEmitted = null;
		this.lastCommitAt = 0;
		this.lastPushAt = 0;
		this.currentLabel = '';
		this.labelStartAt = 0;
	}

	/** Feed one raw frame prediction. Returns a committed char or null. */
	push(label: string, conf: number, now = performance.now()): string | null {
		const windowSize = this.opts.windowSize ?? 5;
		const minConf = this.opts.minConfidence ?? 0.24;
		const cooldown = this.opts.cooldownMs ?? 450;
		const baseHold = this.opts.holdMs ?? 700;

		// Long pause (hand lowered / next sign started late): restart the hold
		// window but keep the cooldown. Short pauses keep their progress.
		if (this.lastPushAt && now - this.lastPushAt > 700) {
			this.currentLabel = '';
			this.labelStartAt = 0;
			this.window = [];
			this.progress = 0;
		}
		this.lastPushAt = now;
		if (!label || conf < 0.16) return null;

		this.window.push({ label, conf, t: now });
		while (this.window.length > windowSize) this.window.shift();

		// Weighted vote: the newest frame counts the most, older frames fade.
		const votes = new Map<string, { w: number; sum: number; n: number }>();
		let totalW = 0;
		for (let i = 0; i < this.window.length; i++) {
			const e = this.window[i];
			const weight = 1 + (i / Math.max(1, this.window.length - 1)) * 1.2;
			const v = votes.get(e.label) ?? { w: 0, sum: 0, n: 0 };
			v.w += weight;
			v.sum += e.conf;
			v.n += 1;
			votes.set(e.label, v);
			totalW += weight;
		}
		let topLabel = '';
		let topW = 0;
		let topAvg = 0;
		for (const [k, v] of votes) {
			if (v.w > topW) {
				topLabel = k;
				topW = v.w;
				topAvg = v.sum / Math.max(1, v.n);
			}
		}
		const topShare = totalW > 0 ? topW / totalW : 0;

		// Switch fast: two matching newest frames are enough, or a clear
		// weighted win over the old label. One noisy frame cannot switch.
		const lastTwo = this.window.slice(-2);
		const streak =
			lastTwo.length === 2 && lastTwo[0].label === topLabel && lastTwo[1].label === topLabel;
		if (topLabel !== this.currentLabel) {
			const currentW = votes.get(this.currentLabel)?.w ?? 0;
			if (!streak && topW <= currentW * 1.35) return null;
			this.currentLabel = topLabel;
			this.labelStartAt = now;
			this.progress = 0;
		}

		const heldMs = now - this.labelStartAt;
		// Adaptive hold: ~350 ms for very confident signs, up to ~800 ms for
		// marginal ones. This keeps typing quick without letting garbage lock.
		const confNorm = Math.max(0, Math.min(1, (topAvg - 0.25) / 0.55));
		const holdMs = Math.round(baseHold * (1.15 - 0.65 * confNorm));
		this.progress = Math.min(1, heldMs / Math.max(1, holdMs));

		// Two letters neck and neck are ambiguous: never commit on flicker.
		if (topShare < 0.58) return null;

		if (
			this.currentLabel &&
			topAvg >= minConf &&
			heldMs >= holdMs &&
			now - this.lastCommitAt >= cooldown
		) {
			// Repeating the same letter needs a deliberate longer hold, so a
			// slow sign change never doubles a character by accident.
			const repeat = this.lastEmitted?.label === this.currentLabel;
			if (repeat && now - this.lastCommitAt < Math.max(cooldown, 900)) return null;
			const committedLabel = this.currentLabel;
			this.lastCommitAt = now;
			this.lastEmitted = { label: committedLabel, conf: topAvg };
			this.window = [];
			this.progress = 0;
			this.currentLabel = '';
			this.labelStartAt = 0;
			return committedLabel;
		}
		return null;
	}
}
