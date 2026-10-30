import { detectZ, detectJ, trailSegments } from '../src/lib/motionLetters.ts';

function makeTrail(points, startT = 0) {
	return points.map(([x, y], i) => ({ x, y, t: startT + i * 40 }));
}

// Z: right, down-left, right (raw camera coords).
const z = makeTrail([
	[0.3, 0.3], [0.34, 0.3], [0.38, 0.3], [0.42, 0.3], [0.46, 0.3],
	[0.43, 0.33], [0.40, 0.36], [0.37, 0.39], [0.34, 0.42],
	[0.38, 0.43], [0.42, 0.43], [0.46, 0.44], [0.50, 0.44]
]);
// J: down then left hook.
const j = makeTrail([
	[0.5, 0.2], [0.5, 0.25], [0.5, 0.3], [0.5, 0.35], [0.5, 0.4],
	[0.47, 0.41], [0.44, 0.42], [0.4, 0.42], [0.37, 0.42]
]);
// Negative: straight horizontal line.
const h = makeTrail([
	[0.3, 0.3], [0.35, 0.3], [0.4, 0.3], [0.45, 0.3], [0.5, 0.3],
	[0.55, 0.3], [0.6, 0.3], [0.65, 0.3]
]);
// Negative: straight down.
const d = makeTrail([
	[0.5, 0.2], [0.5, 0.25], [0.5, 0.3], [0.5, 0.35], [0.5, 0.4],
	[0.5, 0.45], [0.5, 0.5], [0.5, 0.55]
]);
// Negative: tiny jitter.
const n = makeTrail([
	[0.5, 0.5], [0.502, 0.501], [0.5, 0.499], [0.501, 0.5], [0.5, 0.5],
	[0.502, 0.5], [0.5, 0.501], [0.501, 0.5], [0.5, 0.5], [0.501, 0.501]
]);

// Mirrored variants (user signs in a mirror view / other hand).
const zM = makeTrail([
	[0.5, 0.3], [0.46, 0.3], [0.42, 0.3], [0.38, 0.3], [0.34, 0.3],
	[0.37, 0.33], [0.40, 0.36], [0.43, 0.39], [0.46, 0.42],
	[0.42, 0.43], [0.38, 0.43], [0.34, 0.44], [0.30, 0.44]
]);
const jM = makeTrail([
	[0.5, 0.2], [0.5, 0.25], [0.5, 0.3], [0.5, 0.35], [0.5, 0.4],
	[0.53, 0.41], [0.56, 0.42], [0.6, 0.42], [0.63, 0.42]
]);

const results = {
	z_segments: trailSegments(z).map((s) => [s.dx.toFixed(3), s.dy.toFixed(3)]),
	j_segments: trailSegments(j).map((s) => [s.dx.toFixed(3), s.dy.toFixed(3)]),
	z_detectZ: detectZ(z),
	j_detectJ: detectJ(j),
	zMirror_detectZ: detectZ(zM),
	jMirror_detectJ: detectJ(jM),
	horizontal_detectZ: detectZ(h),
	horizontal_detectJ: detectJ(h),
	down_detectJ: detectJ(d),
	down_detectZ: detectZ(d),
	jitter_detectZ: detectZ(n),
	jitter_detectJ: detectJ(n)
};
console.log(JSON.stringify(results, null, 2));

const ok =
	results.z_detectZ === true &&
	results.j_detectJ === true &&
	results.zMirror_detectZ === true &&
	results.jMirror_detectJ === true &&
	results.horizontal_detectZ === false &&
	results.horizontal_detectJ === false &&
	results.down_detectJ === false &&
	results.down_detectZ === false &&
	results.jitter_detectZ === false &&
	results.jitter_detectJ === false;
console.log(ok ? 'ALL PASS' : 'FAIL');
process.exit(ok ? 0 : 1);

