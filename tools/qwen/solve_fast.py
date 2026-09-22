#!/usr/bin/env python3
"""Fast Aliyun slide-captcha target solver.

Input : /tmp/qwen_cap.json {"bg":b64,"piece":b64,
        "bgRect":[x,y,w,h],"pieceRect":[x,y,w,h]}
Output: prints JSON {"L_star","D_est","px0","dx","dy","method","score"}

Handles both observed challenge styles:
- white ghost piece baked into back.png  -> whiteness-blob detection
- dark hole cut out of back.png          -> masked NCC of the piece sprite
"""
import base64
import io
import json
import sys

import numpy as np
from PIL import Image

# piece-left(D) calibration for the 300px embed track
A, B = 0.0035503, 0.0769223


def largest_component(mask):
    from collections import deque
    H, W = mask.shape
    lbl = np.zeros(mask.shape, dtype=int)
    best, bestn = None, 0
    cur = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lbl[y, x] == 0:
                cur += 1
                q = deque([(y, x)])
                lbl[y, x] = cur
                n = 0
                x0 = x1 = x
                y0 = y1 = y
                while q:
                    cy, cx = q.popleft()
                    n += 1
                    x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lbl[ny, nx] == 0:
                            lbl[ny, nx] = cur
                            q.append((ny, nx))
                if n > bestn:
                    bestn, best = n, (x0, x1, y0, y1, n)
    return best


def masked_ncc(gray_b, gray_p, mask):
    ph, pw = gray_p.shape
    bh, bw = gray_b.shape
    m = mask.astype(float)
    n = m.sum()
    sp = (gray_p * m).sum()
    sp2 = (gray_p * gray_p * m).sum()
    var_p = max(sp2 - sp * sp / n, 1e-9)
    out = []
    for by in range(0, bh - ph + 1):
        for bx in range(0, bw - pw + 1):
            win = gray_b[by:by + ph, bx:bx + pw]
            sw = (win * m).sum()
            sw2 = (win * win * m).sum()
            var_w = max(sw2 - sw * sw / n, 1e-9)
            num = (win * gray_p * m).sum() - sp * sw / n
            out.append((num / np.sqrt(var_w * var_p), bx, by))
    out.sort(reverse=True)
    return out


def b64_of(src):
    return src.split(',', 1)[1] if src.startswith('data:') else src


def main():
    d = json.load(open('/tmp/qwen_cap.json'))
    bg = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64_of(d['bg'])))).convert('RGB'), float)
    pc = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64_of(d['piece'])))).convert('RGBA'), float)
    bgr, pcr = d['bgRect'], d['pieceRect']
    sx = bgr[2] / bg.shape[1]          # bg displayed/natural scale (300/296)
    sy = bgr[3] / bg.shape[0]

    alpha = pc[:, :, 3]
    pys, pxs = np.where(alpha > 128)
    px0, py0 = int(pxs.min()), int(pys.min())
    pw, ph = int(pxs.max() - pxs.min() + 1), int(pys.max() - pys.min() + 1)
    patch = pc[py0:py0 + ph, px0:px0 + pw, :3]
    pmask = alpha[py0:py0 + ph, px0:px0 + pw] > 128
    gray_p = patch.mean(axis=2)
    gray_b = bg.mean(axis=2)

    # style 1: white ghost blob
    R, G, Bl = bg[:, :, 0], bg[:, :, 1], bg[:, :, 2]
    mx = np.maximum(np.maximum(R, G), Bl)
    mn = np.minimum(np.minimum(R, G), Bl)
    white = (mx > 150) & ((mx - mn) < 40)
    white[py0 + 3:py0 + ph - 3, px0 + 3:px0 + pw - 3] = False  # ignore piece start area
    comp = largest_component(white)

    # style 2: NCC dark hole
    top = masked_ncc(gray_b, gray_p, pmask)
    score, ndx, ndy = top[0]
    mean = float(np.mean([t[0] for t in top]))

    result = {'px0': px0, 'py0': py0, 'pw': pw, 'ph': ph,
              'ncc_best': [round(score, 4), ndx, ndy], 'ncc_mean': round(mean, 4),
              'scale': [round(sx, 5), round(sy, 5)]}

    ghost = None
    if comp and comp[4] > 200:
        x0, x1, y0, y1, n = comp
        ghost = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
    result['ghost_blob'] = ghost

    # pick the method
    if score >= 0.6:
        dx, dy, method = ndx, ndy, 'ncc'
    elif ghost and abs(ghost[2] - pw) < 14 and abs(ghost[3] - ph) < 14:
        dx, dy, method = ghost[0], ghost[1], 'ghost'
    elif score >= 0.35:
        dx, dy, method = ndx, ndy, 'ncc-weak'
    elif ghost:
        dx, dy, method = ghost[0], ghost[1], 'ghost-weak'
    else:
        print(json.dumps({'error': 'no target found', **result}))
        sys.exit(1)

    # sanity: ghost width should roughly match content width
    L_star = dx * sx - px0
    D = (-B + (B * B + 4 * A * L_star) ** 0.5) / (2 * A)
    result.update({'method': method, 'dx': int(dx), 'dy': int(dy),
                   'L_star': round(float(L_star), 2), 'D_est': round(float(D), 1),
                   'target_piece_page_x': round(bgr[0] + L_star, 2),
                   'grab': [round(pcr[0] + pcr[2] / 2, 1), round(pcr[1] + pcr[3] / 2, 1)]})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
