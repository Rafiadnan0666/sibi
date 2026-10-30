// Smoother timing test with a synthetic clock. Run: node smoother.test.mjs
const { PredictionSmoother } = await import('../src/lib/predictionSmoother.ts');

function sim(seq, opts = {}) {
	const s = new PredictionSmoother({ minConfidence: 0.3, cooldownMs: 450, holdMs: 700, ...opts });
	const events = [];
	let t = 0;
	for (const [label, conf, dt = 100] of seq) {
		t += dt;
		const c = s.push(label, conf, t);
		if (c) events.push({ t, c });
	}
	return { events, progress: s.progress };
}

// 1. Confident hold A for 2s: first commit should be ~<600ms.
const holdA = sim(Array.from({ length: 20 }, () => ['A', 0.85, 100]));
console.log('hold A commits:', holdA.events);

// 2. Switch A -> B: B should commit within ~1s of the first B frame.
const switchAB = sim([
	...Array.from({ length: 10 }, () => ['A', 0.85, 100]),
	...Array.from({ length: 10 }, () => ['B', 0.85, 100])
]);
console.log('switch A->B commits:', switchAB.events);

// 3. Flicker A/B must not commit.
const flicker = sim(Array.from({ length: 24 }, (_, i) => [i % 2 ? 'A' : 'B', 0.7, 100]));
console.log('flicker commits:', flicker.events);

// 4. Medium confidence C should still commit, just a bit slower.
const medium = sim(Array.from({ length: 20 }, () => ['C', 0.42, 100]));
console.log('medium C commits:', medium.events);

// 5. Short speech pause (600ms) mid-hold must not lose progress.
const pause = sim([
	...Array.from({ length: 3 }, () => ['D', 0.8, 100]),
	['D', 0.8, 600],
	...Array.from({ length: 6 }, () => ['D', 0.8, 100])
]);
console.log('pause D commits:', pause.events);

const ok =
	holdA.events.length > 0 &&
	holdA.events[0].t <= 600 &&
	switchAB.events.some((e) => e.c === 'B' && e.t <= 2100) &&
	flicker.events.length === 0 &&
	medium.events.length > 0 &&
	pause.events.length > 0;
console.log(ok ? 'ALL PASS' : 'FAIL');
process.exit(ok ? 0 : 1);

