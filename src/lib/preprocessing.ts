// Image preprocessing for the SIBI CNN (Pengolahan Citra material).
// Mirrors python/train_sibi.py: ROI crop -> resize -> normalize (-> grayscale opt).

import { MODEL_INPUT_SIZE } from './sibiLabels';

export interface PreprocessOptions {
	size?: number;
	normalize?: 'zero-one' | 'neg-one-one';
	grayscale?: boolean;
}

/** Square-crop a pixel-space box from a video frame, with padding + clamping. */
export function cropROI(
	source: HTMLVideoElement,
	box: { x: number; y: number; w: number; h: number }
): HTMLCanvasElement {
	const vw = source.videoWidth || 640;
	const vh = source.videoHeight || 480;
	const side = Math.max(24, Math.max(box.w, box.h) * 1.35);
	const cx = box.x + box.w / 2;
	const cy = box.y + box.h / 2;
	let sx = Math.round(cx - side / 2);
	let sy = Math.round(cy - side / 2);
	let s = Math.round(side);
	sx = Math.max(0, Math.min(vw - 1, sx));
	sy = Math.max(0, Math.min(vh - 1, sy));
	s = Math.max(1, Math.min(vw - sx, vh - sy, s));
	const c = document.createElement('canvas');
	c.width = s;
	c.height = s;
	c.getContext('2d')!.drawImage(source, sx, sy, s, s, 0, 0, s, s);
	return c;
}

/** Resize + normalize a frame ROI to a Float32Array tensor (HWC, RGB). */
export function preprocessToTensor(
	canvas: HTMLCanvasElement,
	opts: PreprocessOptions = {}
): { data: Float32Array; width: number; height: number } {
	const size = opts.size ?? MODEL_INPUT_SIZE;
	const off = document.createElement('canvas');
	off.width = size;
	off.height = size;
	const ctx = off.getContext('2d', { willReadFrequently: true })!;
	ctx.drawImage(canvas, 0, 0, size, size);
	const img = ctx.getImageData(0, 0, size, size);
	const data = new Float32Array(size * size * 3);
	const mode = opts.normalize ?? 'neg-one-one';
	for (let i = 0; i < size * size; i++) {
		let r = img.data[i * 4];
		let g = img.data[i * 4 + 1];
		let b = img.data[i * 4 + 2];
		if (opts.grayscale) {
			const y = 0.299 * r + 0.587 * g + 0.114 * b;
			r = g = b = y;
		}
		if (mode === 'neg-one-one') {
			data[i * 3] = r / 127.5 - 1;
			data[i * 3 + 1] = g / 127.5 - 1;
			data[i * 3 + 2] = b / 127.5 - 1;
		} else {
			data[i * 3] = r / 255;
			data[i * 3 + 1] = g / 255;
			data[i * 3 + 2] = b / 255;
		}
	}
	return { data, width: size, height: size };
}

/** Draw a normalized preview (for the "preprocessing" debug thumbnail). */
export function drawPreview(
	tensor: { data: Float32Array; width: number; height: number },
	target: HTMLCanvasElement,
	mode: 'neg-one-one' | 'zero-one' = 'neg-one-one'
) {
	target.width = tensor.width;
	target.height = tensor.height;
	const ctx = target.getContext('2d')!;
	const img = ctx.createImageData(tensor.width, tensor.height);
	for (let i = 0; i < tensor.width * tensor.height; i++) {
		let r = tensor.data[i * 3];
		let g = tensor.data[i * 3 + 1];
		let b = tensor.data[i * 3 + 2];
		if (mode === 'neg-one-one') {
			r = (r + 1) * 127.5;
			g = (g + 1) * 127.5;
			b = (b + 1) * 127.5;
		} else {
			r *= 255;
			g *= 255;
			b *= 255;
		}
		img.data[i * 4] = r;
		img.data[i * 4 + 1] = g;
		img.data[i * 4 + 2] = b;
		img.data[i * 4 + 3] = 255;
	}
	ctx.putImageData(img, 0, 0);
}
