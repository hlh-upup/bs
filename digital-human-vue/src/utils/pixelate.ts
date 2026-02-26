// 高质量像素化工具：缩放-放大 + 色彩分档 + 可选 Floyd–Steinberg 抖动
// 使用方式：
//   const out = await pixelateImage(dataURL, { blockSize: 10, levels: 16, dithering: true })

export interface PixelateOptions {
  blockSize?: number // 像素块大小（像素）
  levels?: number // 每通道色阶，8/16/32 建议
  dithering?: boolean // 是否启用 Floyd–Steinberg 抖动
  mimeType?: string // 输出类型，默认 image/png
}

export async function pixelateImage(
  dataURL: string,
  options: PixelateOptions = {},
): Promise<string> {
  const { blockSize = 10, levels = 16, dithering = true, mimeType = 'image/png' } = options
  const img = await loadImage(dataURL)
  const width = img.naturalWidth
  const height = img.naturalHeight

  // 极小图避免过度像素化
  if (Math.min(width, height) < blockSize * 2) return dataURL

  // 原图画布
  const base = document.createElement('canvas')
  base.width = width
  base.height = height
  const bctx = base.getContext('2d')!
  bctx.drawImage(img, 0, 0)

  // 缩小画布（先聚合像素）
  const sw = Math.max(1, Math.ceil(width / blockSize))
  const sh = Math.max(1, Math.ceil(height / blockSize))
  const small = document.createElement('canvas')
  small.width = sw
  small.height = sh
  const sctx = small.getContext('2d')!
  sctx.drawImage(base, 0, 0, sw, sh)

  // 对缩小后的像素做色彩分档 + 可选抖动
  let imgData = sctx.getImageData(0, 0, sw, sh)
  imgData = dithering ? fsDither(imgData, levels) : posterizeImageData(imgData, levels)
  sctx.putImageData(imgData, 0, 0)

  // 关闭平滑放大回原尺寸，得到“像素块”
  bctx.clearRect(0, 0, width, height)
  bctx.imageSmoothingEnabled = false
  bctx.drawImage(small, 0, 0, sw, sh, 0, 0, width, height)

  return base.toDataURL(mimeType)
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

// 简单色彩分档（posterize）：将每通道映射到有限档位
function posterizeImageData(imgData: ImageData, levels = 16): ImageData {
  const data = imgData.data
  const step = 255 / Math.max(1, levels - 1)
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.round(data[i] / step) * step // R
    data[i + 1] = Math.round(data[i + 1] / step) * step // G
    data[i + 2] = Math.round(data[i + 2] / step) * step // B
    // A 不变
  }
  return imgData
}

// Floyd–Steinberg 抖动量化（基于 levels 档位）
function fsDither(imgData: ImageData, levels = 16): ImageData {
  const w = imgData.width
  const h = imgData.height
  const d = imgData.data
  const step = 255 / Math.max(1, levels - 1)
  const idx = (x: number, y: number) => (y * w + x) * 4
  const clamp = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v)

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = idx(x, y)
      for (let c = 0; c < 3; c++) {
        const oldv = d[i + c]
        const newv = Math.round(oldv / step) * step
        d[i + c] = newv
        const err = oldv - newv
        // 扩散误差（FS 核）
        if (x + 1 < w) d[i + 4 + c] = clamp(d[i + 4 + c] + (err * 7) / 16)
        if (x - 1 >= 0 && y + 1 < h) d[idx(x - 1, y + 1) + c] = clamp(d[idx(x - 1, y + 1) + c] + (err * 3) / 16)
        if (y + 1 < h) d[idx(x, y + 1) + c] = clamp(d[idx(x, y + 1) + c] + (err * 5) / 16)
        if (x + 1 < w && y + 1 < h) d[idx(x + 1, y + 1) + c] = clamp(d[idx(x + 1, y + 1) + c] + (err * 1) / 16)
      }
    }
  }
  return imgData
}
