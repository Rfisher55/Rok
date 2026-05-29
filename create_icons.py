#!/usr/bin/env python3
"""
Generate PWA icons for ROK Stock Intelligence app.
Pure Python - no external libraries required.
"""
import struct
import zlib
import math
import os

# Colors
BG = (2, 4, 8)          # #020408 - very dark navy
GOLD = (255, 215, 0)     # #FFD700 - gold
GOLD_DIM = (180, 150, 0) # Dimmer gold for inner glow
BLACK = (0, 0, 0)


def make_png(width, height, pixels):
    """Create a valid PNG file from a list of (R,G,B) tuples, row by row."""
    def chunk(name, data):
        c = struct.pack('>I', len(data)) + name + data
        return c + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # 8-bit RGB

    raw = b''
    for row_idx in range(height):
        row_start = row_idx * width
        row = pixels[row_start:row_start + width]
        raw += b'\x00' + b''.join(struct.pack('BBB', r, g, b) for r, g, b in row)

    idat_data = zlib.compress(raw, 9)

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', ihdr)
    png += chunk(b'IDAT', idat_data)
    png += chunk(b'IEND', b'')
    return png


def lerp_color(c1, c2, t):
    """Linearly interpolate between two colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def blend(base, overlay, alpha):
    """Alpha-blend overlay onto base. alpha in [0,1]."""
    return (
        int(base[0] * (1 - alpha) + overlay[0] * alpha),
        int(base[1] * (1 - alpha) + overlay[1] * alpha),
        int(base[2] * (1 - alpha) + overlay[2] * alpha),
    )


def draw_circle_ring(pixels, width, height, cx, cy, radius, thickness, color):
    """Draw an anti-aliased circular ring."""
    r_outer = radius + thickness / 2
    r_inner = radius - thickness / 2

    for py in range(height):
        for px in range(width):
            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx * dx + dy * dy)

            # Anti-aliasing: fade over 1 pixel at edges
            if r_inner - 1 <= dist <= r_outer + 1:
                alpha = 1.0
                if dist < r_inner:
                    alpha = dist - (r_inner - 1)
                elif dist > r_outer:
                    alpha = (r_outer + 1) - dist
                alpha = max(0.0, min(1.0, alpha))

                idx = py * width + px
                pixels[idx] = blend(pixels[idx], color, alpha)


def draw_glow(pixels, width, height, cx, cy, radius, color):
    """Draw a soft radial glow."""
    for py in range(height):
        for px in range(width):
            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radius:
                # Gaussian-like falloff
                t = 1.0 - (dist / radius)
                alpha = t * t * 0.35  # subtle
                idx = py * width + px
                pixels[idx] = blend(pixels[idx], color, alpha)


def draw_pixel_rect(pixels, width, px, py, pw, ph, color):
    """Draw a filled rectangle."""
    for ry in range(ph):
        for rx in range(pw):
            idx = (py + ry) * width + (px + rx)
            if 0 <= idx < len(pixels):
                pixels[idx] = color


def draw_letter_R(pixels, width, ox, oy, scale, color):
    """Draw block-pixel letter R at offset (ox,oy) with given scale."""
    # R pattern on a 5-wide x 7-tall grid
    pattern = [
        "XXX..",
        "X..X.",
        "X..X.",
        "XXX..",
        "XX...",
        "X.X..",
        "X..X.",
    ]
    for row_i, row in enumerate(pattern):
        for col_i, ch in enumerate(row):
            if ch == 'X':
                draw_pixel_rect(
                    pixels, width,
                    ox + col_i * scale, oy + row_i * scale,
                    scale, scale, color
                )


def draw_letter_O(pixels, width, ox, oy, scale, color):
    """Draw block-pixel letter O at offset (ox,oy) with given scale."""
    pattern = [
        ".XXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ]
    for row_i, row in enumerate(pattern):
        for col_i, ch in enumerate(row):
            if ch == 'X':
                draw_pixel_rect(
                    pixels, width,
                    ox + col_i * scale, oy + row_i * scale,
                    scale, scale, color
                )


def draw_letter_K(pixels, width, ox, oy, scale, color):
    """Draw block-pixel letter K at offset (ox,oy) with given scale."""
    pattern = [
        "X...X",
        "X..X.",
        "X.X..",
        "XX...",
        "X.X..",
        "X..X.",
        "X...X",
    ]
    for row_i, row in enumerate(pattern):
        for col_i, ch in enumerate(row):
            if ch == 'X':
                draw_pixel_rect(
                    pixels, width,
                    ox + col_i * scale, oy + row_i * scale,
                    scale, scale, color
                )


def create_icon(size):
    """Create a square icon of the given size. Returns PNG bytes."""
    width = height = size
    pixels = [BG] * (width * height)

    cx = width // 2
    cy = height // 2

    # Scale factor relative to 192
    sf = size / 192.0

    # 1. Draw central glow
    glow_radius = int(70 * sf)
    draw_glow(pixels, width, height, cx, cy, glow_radius, GOLD)

    # 2. Draw gold ring border
    ring_radius = int(85 * sf)
    ring_thickness = max(4, int(8 * sf))
    draw_circle_ring(pixels, width, height, cx, cy, ring_radius, ring_thickness, GOLD)

    # 3. Draw "ROK" text centered
    # Each letter is 5 cols x 7 rows of pixels, at `scale` px per pixel-cell
    # With 2-pixel gap between letters
    scale = max(2, int(round(6 * sf)))
    gap = max(1, int(round(3 * sf)))

    letter_w = 5 * scale  # width per letter
    letter_h = 7 * scale  # height of letters
    total_w = 3 * letter_w + 2 * gap  # R + gap + O + gap + K

    text_x = cx - total_w // 2
    text_y = cy - letter_h // 2

    draw_letter_R(pixels, width, text_x, text_y, scale, GOLD)
    draw_letter_O(pixels, width, text_x + letter_w + gap, text_y, scale, GOLD)
    draw_letter_K(pixels, width, text_x + 2 * (letter_w + gap), text_y, scale, GOLD)

    # 4. Small gold center dot beneath text for decoration
    dot_r = max(2, int(4 * sf))
    dot_y = text_y + letter_h + max(4, int(8 * sf))
    for py in range(height):
        for px in range(width):
            dx = px - cx
            dy = py - dot_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= dot_r:
                alpha = max(0.0, min(1.0, (dot_r - dist) / max(1, dot_r * 0.5)))
                idx = py * width + px
                pixels[idx] = blend(pixels[idx], GOLD, alpha)

    return make_png(width, height, pixels)


def main():
    out_dir = "/home/user/Rok/docs"
    os.makedirs(out_dir, exist_ok=True)

    for size in (192, 512):
        print(f"Generating {size}x{size} icon...")
        png_data = create_icon(size)
        path = os.path.join(out_dir, f"icon-{size}.png")
        with open(path, 'wb') as f:
            f.write(png_data)
        print(f"  Written: {path} ({len(png_data):,} bytes)")

    # Quick validation: check PNG signature and IHDR chunk for each file
    for size in (192, 512):
        path = os.path.join(out_dir, f"icon-{size}.png")
        with open(path, 'rb') as f:
            data = f.read()
        sig = data[:8]
        assert sig == b'\x89PNG\r\n\x1a\n', f"Bad PNG signature in {path}"
        # IHDR length
        ihdr_len = struct.unpack('>I', data[8:12])[0]
        assert ihdr_len == 13, f"Unexpected IHDR length {ihdr_len}"
        w, h = struct.unpack('>II', data[16:24])
        assert w == size and h == size, f"Wrong dimensions {w}x{h} for {size}"
        print(f"  Validated: {path} — {w}x{h}, {len(data):,} bytes — OK")

    print("\nAll icons generated successfully.")


if __name__ == '__main__':
    main()
