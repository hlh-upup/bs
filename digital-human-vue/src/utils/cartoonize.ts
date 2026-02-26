// 二次元风格卡通化：平滑(近似双边) + 色彩量化 + 边缘检测 + 叠加
// 用法：
//  const out = await cartoonizeImage(dataURL, { levels: 12, edgeThreshold: 80, blurPasses: 2 })

export interface CartoonizeOptions {
  levels?: number // 每通道色阶（建议 6~16）
  edgeThreshold?: number // 边缘阈值（0~255）
  blurPasses?: number // 平滑次数（1~4，次数越多越“干净”）
  mimeType?: string
}

export async function cartoonizeImage(
  dataURL: string,
  options: CartoonizeOptions = {},
): Promise<string> {
  const { levels = 12, edgeThreshold = 80, blurPasses = 2, mimeType = 'image/png' } = options
  const img = await loadImage(dataURL)
  const w = img.naturalWidth
  const h = img.naturalHeight

  if (w * h === 0) return dataURL

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0)

  // 1) 读取像素
  let imgData = ctx.getImageData(0, 0, w, h)

  // 2) 近似双边：用多次轻度盒滤平滑（保边效果不如真正双边，但性能更好）
  for (let i = 0; i < blurPasses; i++) {
    imgData = boxBlur(imgData, w, h, 1)
  }

  // 3) 色彩量化（posterize）
  posterize(imgData.data, levels)

  // 4) 边缘检测（Sobel + 灰度）
  const edges = sobelEdges(imgData, w, h)

  // 5) 将量化图写回
  ctx.putImageData(imgData, 0, 0)

  // 6) 叠加线稿（根据阈值，将强边缘画成黑线）
  const edgeImg = ctx.getImageData(0, 0, w, h)
  const ed = edgeImg.data
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4
      if (edges[i] >= edgeThreshold) {
        // 拉黑线条
        ed[i] = 0
        ed[i + 1] = 0
        ed[i + 2] = 0
        // alpha 保持不变
      }
    }
  }
  ctx.putImageData(edgeImg, 0, 0)

  return canvas.toDataURL(mimeType)
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

// 简单盒滤（半径 1，分离卷积）
function boxBlur(imgData: ImageData, w: number, h: number, radius = 1): ImageData {
  const src = imgData.data
  const out = new Uint8ClampedArray(src.length)
  const tmp = new Uint8ClampedArray(src.length)
  const channels = 4

  // 横向
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * channels
      for (let c = 0; c < 3; c++) {
        let sum = 0
        let count = 0
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx
          if (xx >= 0 && xx < w) {
            sum += src[(y * w + xx) * channels + c]
            count++
          }
        }
        tmp[i + c] = sum / count
      }
      tmp[i + 3] = src[i + 3]
    }
  }

  // 纵向
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * channels
      for (let c = 0; c < 3; c++) {
        let sum = 0
        let count = 0
        for (let dy = -radius; dy <= radius; dy++) {
          const yy = y + dy
          if (yy >= 0 && yy < h) {
            sum += tmp[(yy * w + x) * channels + c]
            count++
          }
        }
        out[i + c] = sum / count
      }
      out[i + 3] = tmp[i + 3]
    }
  }

  return new ImageData(out, w, h)
}

function posterize(data: Uint8ClampedArray, levels: number) {
  const step = 255 / Math.max(1, levels - 1)
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.round(data[i] / step) * step
    data[i + 1] = Math.round(data[i + 1] / step) * step
    data[i + 2] = Math.round(data[i + 2] / step) * step
  }
}

function sobelEdges(imgData: ImageData, w: number, h: number): Uint8ClampedArray {
  // 转灰度
  const g = new Uint8ClampedArray(w * h)
  const d = imgData.data
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    // 亮度加权（BT.601）
    g[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
  }

  const out = new Uint8ClampedArray(w * h * 4)
  const gxK = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
  ]
  const gyK = [
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1,
  ]

  const idx = (x: number, y: number) => y * w + x

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let gx = 0
      let gy = 0
      let k = 0
      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const v = g[idx(x + xx, y + yy)]
          gx += v * gxK[k]
          gy += v * gyK[k]
          k++
        }
      }
      const mag = Math.sqrt(gx * gx + gy * gy)
      const m = mag > 255 ? 255 : mag
      const p = idx(x, y) * 4
      out[p] = m
      out[p + 1] = m
      out[p + 2] = m
      out[p + 3] = 255
    }
  }

  return out
}
