// Verify TF.js graph models against Keras reference probabilities.
import * as tf from '@tensorflow/tfjs';
import { readFileSync } from 'node:fs';

await tf.setBackend('cpu');
await tf.ready();

const BASE = 'http://127.0.0.1:8123/models';
const vecs = JSON.parse(readFileSync(process.argv[2] ?? '', 'utf8'));

let fails = 0;
async function check(url, samples, shape, nOut, tag) {
	const m = await tf.loadGraphModel(url);
	let correct = 0;
	let maxDiff = 0;
	for (const s of samples) {
		const t = tf.tensor(s.feat, shape);
		const r = await m.executeAsync(t);
		const ten = Array.isArray(r) ? r[0] : r;
		const probs = Array.from(await ten.data());
		t.dispose();
		if (Array.isArray(r)) r.forEach((x) => x.dispose());
		else ten.dispose();
		if (probs.length !== nOut) {
			console.log(tag, 'OUTPUT DIM MISMATCH', probs.length);
			fails++;
			return;
		}
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		for (let i = 0; i < probs.length; i++)
			maxDiff = Math.max(maxDiff, Math.abs(probs[i] - s.probs[i]));
		const pred = vecs.labels[best] ?? `dead-${best}`;
		if (pred === s.label) correct++;
		else if (tag !== 'baseline' || best < 24) {
			/* count */
		}
	}
	console.log(
		`${tag}: acc=${(correct / samples.length).toFixed(4)} maxAbsDiff_vs_keras=${maxDiff.toExponential(2)}`
	);
	if (maxDiff > 2e-2) {
		console.log(tag, 'NUMERIC MISMATCH');
		fails++;
	}
	m.dispose();
}

await check(`${BASE}/sibi-joint/model.json`, vecs.joint, [1, 63], 24, 'joint');
await check(`${BASE}/sibi-joint-baseline/model.json`, vecs.baseline, [1, 63, 1], 26, 'baseline');
// image models: shape/run check (softmax sums to 1)
for (const tag of ['sibi-advanced', 'sibi-image']) {
	const m = await tf.loadGraphModel(`${BASE}/${tag}/model.json`);
	const t = tf.randomNormal([1, 128, 128, 3]);
	const r = await m.executeAsync(t);
	const ten = Array.isArray(r) ? r[0] : r;
	const p = Array.from(await ten.data());
	const sum = p.reduce((a, b) => a + b, 0);
	console.log(`${tag}: outDim=${p.length} softmaxSum=${sum.toFixed(4)}`);
	if (p.length !== 24 || Math.abs(sum - 1) > 1e-3) fails++;
	t.dispose();
	if (Array.isArray(r)) r.forEach((x) => x.dispose());
	else ten.dispose();
	m.dispose();
}
console.log(fails === 0 ? 'VERIFY-OK' : 'VERIFY-FAIL');
process.exit(fails === 0 ? 0 : 1);
