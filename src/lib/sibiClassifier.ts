// SIBI model engine: THREE real trained TF.js graph models.
//
//  1. sibi-joint (PRIMARY)      — MLP on wrist-relative/hand-size normalized
//     21-joint vector. Retrained here: val acc 87.1% (24 classes A-I,K-Y).
//  2. sibi-joint-baseline       — upstream AJustiago/SIBI-Recognition Conv1D
//     on raw pixel joints (MIT). Val acc 74.2%. Kept as pembanding + vote.
//  3. sibi-image                — MobileNetV2 transfer learning on SIBI hand
//     photos (480 train / 240 val). Val acc 73.8%. ROI cross-check.
//
// J and Z are excluded everywhere: they are dynamic SIBI signs (motion),
// not static handshapes — documented limitation, not a bug.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'demo' | 'error';

export interface ModelSlot {
	key: 'joint' | 'baseline' | 'image';
	name: string;
	url: string;
	valAcc: string;
	credit: string;
	status: ModelStatus;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	model: any | null;
	labels: string[];
}

export interface ScoredLetter {
	label: string;
	confidence: number;
}

export interface JointResult extends ScoredLetter {
	probs: number[];
	top3: ScoredLetter[];
	latencyMs: number;
}

export const JOINT_LABELS_FALLBACK = [
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
	'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
	'T', 'U', 'V', 'W', 'X', 'Y'
];

async function fetchLabels(url: string): Promise<string[]> {
	try {
		const r = await fetch(url);
		if (!r.ok) return JOINT_LABELS_FALLBACK;
		const j = (await r.json()) as unknown;
		return Array.isArray(j) && j.length > 0 ? (j as string[]) : JOINT_LABELS_FALLBACK;
	} catch {
		return JOINT_LABELS_FALLBACK;
	}
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function runGraph(model: any, input: tf.Tensor): Promise<Float32Array | null> {
	try {
		const r = await model.executeAsync(input);
		const ten = (Array.isArray(r) ? r[0] : r) as tf.Tensor;
		const data = (await ten.data()) as Float32Array;
		if (Array.isArray(r)) r.forEach((t: tf.Tensor) => t.dispose());
		else ten.dispose();
		return data;
	} catch {
		return null;
	}
}

function top(probs: ArrayLike<number>, labels: string[], k = 3): ScoredLetter[] {
	const idx = Array.from(probs)
		.map((p, i) => i)
		.sort((a, b) => (probs[b] as number) - (probs[a] as number))
		.slice(0, k);
	return idx.map((i) => ({ label: labels[i] ?? '?', confidence: probs[i] as number }));
}

export class SibiEngine {
	slots: ModelSlot[] = [
		{
			key: 'joint', name: 'Sendi (utama)', url: '/models/sibi-joint/model.json',
			valAcc: '87,1%', credit: 'dilati ulang di sini (MLP ternormalisasi)',
			status: 'idle', model: null, labels: []
		},
		{
			key: 'baseline', name: 'Sendi baseline', url: '/models/sibi-joint-baseline/model.json',
			valAcc: '74,2%', credit: 'AJustiago/SIBI-Recognition (MIT)',
			status: 'idle', model: null, labels: []
		},
		{
			key: 'image', name: 'Citra ROI', url: '/models/sibi-image/model.json',
			valAcc: '73,8%', credit: 'MobileNetV2, dilatih di sini (480 foto)',
			status: 'idle', model: null, labels: []
		}
	];

	get readyCount(): number {
		return this.slots.filter((s) => s.status === 'ready').length;
	}

	get primaryReady(): boolean {
		return this.slots[0].status === 'ready';
	}

	async load(): Promise<void> {
		await tf.ready();
		try {
			await tf.setBackend('webgl');
		} catch {
			await tf.setBackend('cpu');
		}
		await Promise.all(
			this.slots.map(async (s) => {
				s.status = 'loading';
				try {
					const labelsUrl = s.url.replace(/model\.json$/, 'labels.json');
					const [m, labels] = await Promise.all([
						tf.loadGraphModel(s.url),
						fetchLabels(labelsUrl)
					]);
					s.model = m;
					s.labels = labels.length >= 24 ? labels.slice(0, 24) : [...JOINT_LABELS_FALLBACK];
					try {
						const warm =
							s.key === 'joint'
								? tf.zeros([1, 63])
								: s.key === 'baseline'
									? tf.zeros([1, 63, 1])
									: tf.zeros([1, 128, 128, 3]);
						const r = await m.executeAsync(warm);
						if (Array.isArray(r)) r.forEach((t: tf.Tensor) => t.dispose());
						else (r as tf.Tensor).dispose();
						warm.dispose();
					} catch {
						// warmup non-fatal
					}
					s.status = 'ready';
				} catch {
					s.status = 'error';
					s.model = null;
				}
			})
		);
	}

	/** Primary: 63-vector normalized joints -> letter + full distribution. */
	async predictJoint(feats: Float32Array): Promise<JointResult | null> {
		const s = this.slots[0];
		if (s.status !== 'ready' || !s.model) return null;
		const t0 = performance.now();
		const input = tf.tensor2d(feats, [1, 63]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		return {
			label: s.labels[best] ?? '?',
			confidence: probs[best],
			probs: Array.from(probs),
			top3: top(probs, s.labels, 3),
			latencyMs: performance.now() - t0
		};
	}

	/** Baseline replica: raw pixel joints (63,1) -> letter (outputs 24,25 dead). */
	async predictBaseline(px: Float32Array): Promise<ScoredLetter | null> {
		const s = this.slots[1];
		if (s.status !== 'ready' || !s.model) return null;
		const input = tf.tensor3d(px, [1, 63, 1]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		if (best >= 24) return { label: '?', confidence: probs[best] };
		return { label: s.labels[best] ?? '?', confidence: probs[best] };
	}

	/** Image cross-check: 128x128x3 [-1,1] ROI tensor -> letter. */
	async predictImage(tensorData: Float32Array): Promise<ScoredLetter | null> {
		const s = this.slots[2];
		if (s.status !== 'ready' || !s.model) return null;
		const input = tf.tensor4d(tensorData, [1, 128, 128, 3]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		return { label: s.labels[best] ?? '?', confidence: probs[best] };
	}

	dispose() {
		for (const s of this.slots) {
			try {
				s.model?.dispose();
			} catch {
				// noop
			}
			s.model = null;
			s.status = 'idle';
		}
	}
}

// --- Temporal smoothing ------------------------------------------------------
// Hold a sign steady: require `minFrames` agreements inside a sliding window,
// a minimum confidence, and a cooldown so one physical sign = one character.

export interface SmootherOptions {
	windowSize?: number;
	minFrames?: number;
	minConfidence?: number;
	cooldownMs?: number;
}

export class PredictionSmoother {
	private window: { label: string; conf: number; t: number }[] = [];
	private lastCommitAt = 0;
	lastEmitted: { label: string; conf: number } | null = null;
	progress = 0; // 0..1 stability progress for the current majority label

	constructor(private opts: SmootherOptions = {}) {}

	reset() {
		this.window = [];
		this.progress = 0;
		this.lastEmitted = null;
		this.lastCommitAt = 0;
	}

	/** Feed one raw frame prediction. Returns a committed char or null. */
	push(label: string, conf: number, now = performance.now()): string | null {
		const windowSize = this.opts.windowSize ?? 12;
		const minFrames = this.opts.minFrames ?? 7;
		const minConf = this.opts.minConfidence ?? 0.55;
		const cooldown = this.opts.cooldownMs ?? 1200;
		this.window.push({ label, conf, t: now });
		if (this.window.length > windowSize) this.window.shift();

		const counts = new Map<string, { n: number; sum: number }>();
		for (const w of this.window) {
			const e = counts.get(w.label) ?? { n: 0, sum: 0 };
			e.n += 1;
			e.sum += w.conf;
			counts.set(w.label, e);
		}
		let topLabel = '';
		let topN = 0;
		let topAvg = 0;
		for (const [k, v] of counts) {
			if (v.n > topN) {
				topLabel = k;
				topN = v.n;
				topAvg = v.sum / v.n;
			}
		}
		this.progress = Math.min(1, topN / minFrames);

		if (topN >= minFrames && topAvg >= minConf && now - this.lastCommitAt >= cooldown) {
			const tail = this.window.slice(-3);
			if (tail.every((w) => w.label === topLabel)) {
				this.lastCommitAt = now;
				this.lastEmitted = { label: topLabel, conf: topAvg };
				this.window = [];
				this.progress = 0;
				return topLabel;
			}
		}
		return null;
	}
}
