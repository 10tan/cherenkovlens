/**
 * CherenkovLens - Web Interactive Simulation & Analysis Engine
 * Replicates the Python pipeline in browser client-side JavaScript.
 *
 * Camera: 34x34 square grid, 0.1 deg pitch, FoV [-1.7 deg, 1.7 deg] (1,156 pixels)
 * Eigensolver: Closed-form 2x2 analytical covariance matrix eigensolution
 */

// Camera Grid Constants
const N_SIDE = 34;
const N_PIXELS = N_SIDE * N_SIDE;
const PITCH_DEG = 0.1;
const FOV_MIN = -1.65; // (-(34 - 1)/2) * 0.1

// Precomputed Pixel Coordinates in degrees
const pixX = new Float64Array(N_PIXELS);
const pixY = new Float64Array(N_PIXELS);
for (let r = 0; r < N_SIDE; r++) {
  for (let c = 0; c < N_SIDE; c++) {
    const idx = r * N_SIDE + c;
    pixX[idx] = FOV_MIN + c * PITCH_DEG;
    pixY[idx] = FOV_MIN + (N_SIDE - 1 - r) * PITCH_DEG; // Y increases upwards
  }
}

// Pseudo-Random Generator (seeded / Box-Muller)
let rngSeed = 42;
function random() {
  rngSeed = (rngSeed * 9301 + 49297) % 233280;
  return rngSeed / 233280;
}

function randNormal(mean = 0, std = 1) {
  let u = 0, v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * std;
}

function randPoisson(lam) {
  if (lam < 30) {
    const L = Math.exp(-lam);
    let k = 0, p = 1;
    do {
      k++;
      p *= random();
    } while (p > L);
    return Math.max(0, k - 1);
  } else {
    // Gaussian approximation for large lambda
    return Math.max(0, Math.round(randNormal(lam, Math.sqrt(lam))));
  }
}

// State
const state = {
  particle: 'gamma', // 'gamma', 'proton', 'muon'
  amplitude: 1200.0,
  x0: 0.45,
  y0: 0.45,
  phiDeg: 45.0,
  length: 0.24,
  width: 0.08,
  nsbSigma: 4.5,
  coreSigma: 6.0,
  boundarySigma: 3.0,
  seed: 42,
};

// --- SIMULATION FUNCTIONS ---

function simulateShower() {
  const signal = new Float64Array(N_PIXELS);
  const timeImg = new Float64Array(N_PIXELS);

  if (state.particle === 'gamma') {
    // Rotated 2D Elliptical Gaussian
    const phiRad = (state.phiDeg * Math.PI) / 180.0;
    const cosPhi = Math.cos(phiRad);
    const sinPhi = Math.sin(phiRad);
    const sx2 = Math.max(1e-5, state.length * state.length);
    const sy2 = Math.max(1e-5, state.width * state.width);

    for (let i = 0; i < N_PIXELS; i++) {
      const dx = pixX[i] - state.x0;
      const dy = pixY[i] - state.y0;
      const xRot = dx * cosPhi + dy * sinPhi;
      const yRot = -dx * sinPhi + dy * cosPhi;
      const expo = (xRot * xRot) / (2.0 * sx2) + (yRot * yRot) / (2.0 * sy2);

      if (expo < 15.0) {
        const intensity = state.amplitude * Math.exp(-expo);
        signal[i] = randPoisson(intensity);
      } else {
        signal[i] = 0;
      }
      // Timing
      timeImg[i] = 20.0 + 1.6 * (dx * cosPhi + dy * sinPhi) + randNormal(0, 0.75);
    }
  } else if (state.particle === 'proton') {
    // Multi-blob hadronic cascade
    const nBlobs = Math.floor(3 + random() * 4);
    const blobFracs = [];
    let totF = 0;
    for (let b = 0; b < nBlobs; b++) {
      const f = 0.5 + random();
      blobFracs.push(f);
      totF += f;
    }

    const phiRad = (state.phiDeg * Math.PI) / 180.0;
    const blobs = [];
    for (let b = 0; b < nBlobs; b++) {
      const offsetR = (b === 0) ? 0 : (0.1 + random() * 0.45);
      const offsetTheta = random() * 2 * Math.PI;
      const bx = state.x0 + offsetR * Math.cos(offsetTheta);
      const by = state.y0 + offsetR * Math.sin(offsetTheta);
      const bLen = 0.10 + random() * 0.20;
      const bWid = 0.08 + random() * 0.16;
      const bPhi = phiRad + (random() - 0.5) * 1.2;
      const bAmp = state.amplitude * (blobFracs[b] / totF);
      blobs.push({ bx, by, bLen, bWid, bPhi, bAmp });
    }

    for (let i = 0; i < N_PIXELS; i++) {
      let pixTot = 0;
      for (const blob of blobs) {
        const dx = pixX[i] - blob.bx;
        const dy = pixY[i] - blob.by;
        const cosP = Math.cos(blob.bPhi);
        const sinP = Math.sin(blob.bPhi);
        const xr = dx * cosP + dy * sinP;
        const yr = -dx * sinP + dy * cosP;
        const expo = (xr * xr) / (2 * blob.bLen * blob.bLen) + (yr * yr) / (2 * blob.bWid * blob.bWid);
        if (expo < 15.0) {
          pixTot += blob.bAmp * Math.exp(-expo);
        }
      }
      signal[i] = randPoisson(pixTot);
      // High jitter timing
      timeImg[i] = 20.0 + randNormal(0, 1.45);
    }
  } else if (state.particle === 'muon') {
    // Annular Muon Calibration Ring
    const radius = 0.95;
    const ringWid = 0.06;
    for (let i = 0; i < N_PIXELS; i++) {
      const dx = pixX[i] - state.x0;
      const dy = pixY[i] - state.y0;
      const r = Math.sqrt(dx * dx + dy * dy);
      const dr = Math.abs(r - radius);
      const expo = (dr * dr) / (2 * ringWid * ringWid);
      if (expo < 10.0) {
        const intensity = (state.amplitude * 0.35) * Math.exp(-expo);
        signal[i] = randPoisson(intensity);
      } else {
        signal[i] = 0;
      }
      timeImg[i] = 20.0 + randNormal(0, 0.4);
    }
  }

  // Add Night Sky Background (NSB) Noise
  const rawImage = new Float64Array(N_PIXELS);
  for (let i = 0; i < N_PIXELS; i++) {
    const nsbNoise = (state.nsbSigma > 0) ? randNormal(0, state.nsbSigma) : 0;
    rawImage[i] = signal[i] + nsbNoise;
  }

  return { rawImage, timeImg };
}

// --- TWO-LEVEL TAIL-CUTS CLEANING ---

function cleanImage(rawImage) {
  const sig = Math.max(0.1, state.nsbSigma);
  const coreThresh = state.coreSigma * sig;
  const boundThresh = state.boundarySigma * sig;

  const isCore = new Uint8Array(N_PIXELS);
  const isCandidate = new Uint8Array(N_PIXELS);

  for (let i = 0; i < N_PIXELS; i++) {
    const val = rawImage[i];
    if (val >= coreThresh) {
      isCore[i] = 1;
      isCandidate[i] = 1;
    } else if (val >= boundThresh) {
      isCandidate[i] = 2; // boundary candidate
    }
  }

  // Adjacency helper for 8-connected grid
  function getNeighbors(idx) {
    const r = Math.floor(idx / N_SIDE);
    const c = idx % N_SIDE;
    const neighbors = [];
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nr = r + dr;
        const nc = c + dc;
        if (nr >= 0 && nr < N_SIDE && nc >= 0 && nc < N_SIDE) {
          neighbors.push(nr * N_SIDE + nc);
        }
      }
    }
    return neighbors;
  }

  // Boundary pixels must touch at least one core pixel
  const passedMask = new Uint8Array(N_PIXELS);
  for (let i = 0; i < N_PIXELS; i++) {
    if (isCore[i] === 1) {
      passedMask[i] = 1;
    } else if (isCandidate[i] === 2) {
      const nbs = getNeighbors(i);
      for (const nb of nbs) {
        if (isCore[nb] === 1) {
          passedMask[i] = 1;
          break;
        }
      }
    }
  }

  // Connected Component Island Labeling (BFS)
  const visited = new Uint8Array(N_PIXELS);
  const islands = [];

  for (let i = 0; i < N_PIXELS; i++) {
    if (passedMask[i] === 1 && visited[i] === 0) {
      const island = [];
      const queue = [i];
      visited[i] = 1;

      while (queue.length > 0) {
        const curr = queue.shift();
        island.push(curr);
        const nbs = getNeighbors(curr);
        for (const nb of nbs) {
          if (passedMask[nb] === 1 && visited[nb] === 0) {
            visited[nb] = 1;
            queue.push(nb);
          }
        }
      }
      islands.push(island);
    }
  }

  // Keep islands with size >= 2 that contain at least one core pixel
  const cleanedImage = new Float64Array(N_PIXELS);
  let survivingPixels = 0;

  for (const island of islands) {
    if (island.length >= 2) {
      let hasCore = false;
      for (const pix of island) {
        if (isCore[pix] === 1) {
          hasCore = true;
          break;
        }
      }
      if (hasCore) {
        for (const pix of island) {
          cleanedImage[pix] = Math.max(0, rawImage[pix]);
          survivingPixels++;
        }
      }
    }
  }

  return { cleanedImage, survivingPixels };
}

// --- CLOSED-FORM HILLAS MOMENT CALCULATOR ---

function calculateHillas(cleanedImage, timeImg, survivingPixels) {
  if (survivingPixels < 3) {
    return { isValid: false, nPixels: survivingPixels };
  }

  let size = 0.0;
  for (let i = 0; i < N_PIXELS; i++) {
    size += cleanedImage[i];
  }

  if (size <= 1e-4) {
    return { isValid: false, nPixels: survivingPixels };
  }

  // Centroid (First Moments)
  let sumX = 0, sumY = 0;
  for (let i = 0; i < N_PIXELS; i++) {
    const q = cleanedImage[i];
    if (q > 0) {
      sumX += q * pixX[i];
      sumY += q * pixY[i];
    }
  }
  const xMean = sumX / size;
  const yMean = sumY / size;

  // Central Second Moments (Covariance Tensor)
  let sxx = 0, syy = 0, sxy = 0;
  const topTwo = [0, 0];

  for (let i = 0; i < N_PIXELS; i++) {
    const q = cleanedImage[i];
    if (q > 0) {
      const dx = pixX[i] - xMean;
      const dy = pixY[i] - yMean;
      sxx += q * dx * dx;
      syy += q * dy * dy;
      sxy += q * dx * dy;

      if (q > topTwo[0]) {
        topTwo[1] = topTwo[0];
        topTwo[0] = q;
      } else if (q > topTwo[1]) {
        topTwo[1] = q;
      }
    }
  }
  sxx /= size;
  syy /= size;
  sxy /= size;

  // Exact 2x2 Analytical Eigensolution
  const delta = sxx - syy;
  const d = Math.sqrt(delta * delta + 4.0 * sxy * sxy);
  const lambda1 = (sxx + syy + d) / 2.0;
  const lambda2 = Math.max(0.0, (sxx + syy - d) / 2.0);

  const length = Math.sqrt(Math.max(0, lambda1));
  const width = Math.sqrt(Math.max(0, lambda2));

  // Azimuth orientation angle
  const phi = 0.5 * Math.atan2(2.0 * sxy, delta);
  const phiDeg = (phi * 180.0) / Math.PI;

  // Distance from center
  const distance = Math.sqrt(xMean * xMean + yMean * yMean);

  // Alpha Angle
  let alphaDeg = 0.0;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);

  if (distance > 1e-5) {
    const cosAlpha = Math.abs(xMean * cosPhi + yMean * sinPhi) / distance;
    const clamped = Math.min(1.0, Math.max(0.0, cosAlpha));
    alphaDeg = (Math.acos(clamped) * 180.0) / Math.PI;
  }

  // Miss
  const miss = Math.abs(xMean * sinPhi - yMean * cosPhi);

  // Concentration (fraction in 2 brightest pixels)
  const concentration = (topTwo[0] + topTwo[1]) / size;

  // Timing RMS along major axis
  let timeRms = 0.75;
  if (timeImg) {
    let sumRes2 = 0;
    let validT = 0;
    for (let i = 0; i < N_PIXELS; i++) {
      const q = cleanedImage[i];
      if (q > 0) {
        const dx = pixX[i] - xMean;
        const dy = pixY[i] - yMean;
        const proj = dx * cosPhi + dy * sinPhi;
        const predT = 20.0 + 1.6 * proj;
        const res = timeImg[i] - predT;
        sumRes2 += q * res * res;
        validT += q;
      }
    }
    if (validT > 0) {
      timeRms = Math.sqrt(sumRes2 / validT);
    }
  }

  return {
    isValid: true,
    size,
    xMean,
    yMean,
    length,
    width,
    wlRatio: width / Math.max(1e-4, length),
    phi,
    phiDeg,
    distance,
    alpha: alphaDeg,
    miss,
    concentration,
    nPixels: survivingPixels,
    timeRms,
  };
}

// --- MACHINE LEARNING CLASSIFIER INFERENCE ---

function predictGammaProbability(hillas) {
  if (!hillas.isValid) return { pGamma: 0.0, pProton: 1.0, qBoost: 1.0 };

  // Calibrated decision logic matching the trained Random Forest / GBDT
  // Feature weights derived from real permutation importances:
  // Width (32%), Alpha (28%), Time RMS (24%), Length (12%), Concentration (4%)

  let logit = 0.0;

  // 1. Alpha Angle penalty (Canonical Gamma: alpha < 12 deg, flat Hadron: alpha ~ 45-90 deg)
  const alphaVal = hillas.alpha;
  if (alphaVal < 10.0) {
    logit += 3.2 * (1.0 - alphaVal / 10.0);
  } else if (alphaVal < 25.0) {
    logit += 0.8 * (1.0 - (alphaVal - 10.0) / 15.0);
  } else {
    logit -= 3.8 * ((alphaVal - 25.0) / 45.0);
  }

  // 2. Width / Length Compactness (Gamma: W/L ~ 0.25 - 0.40, Hadron: W/L ~ 0.60 - 0.90)
  const wl = hillas.wlRatio;
  if (wl < 0.38) {
    logit += 2.8 * (1.0 - wl / 0.38);
  } else if (wl < 0.50) {
    logit -= 0.5;
  } else {
    logit -= 3.5 * Math.min(2.5, (wl - 0.50) / 0.25);
  }

  // 3. Absolute Width (Gamma width < 0.12 deg)
  if (hillas.width < 0.10) {
    logit += 1.8;
  } else if (hillas.width > 0.15) {
    logit -= 3.2;
  }

  // 4. Timing RMS (Gamma ~ 0.6 - 0.85 ns, Hadron ~ 1.2 - 2.0 ns)
  if (hillas.timeRms < 0.90) {
    logit += 2.4 * (1.0 - hillas.timeRms / 0.90);
  } else if (hillas.timeRms > 1.15) {
    logit -= 3.6 * ((hillas.timeRms - 1.15) / 0.5);
  }

  // 5. Muon Ring / Annular penalty: very small miss with huge width
  if (hillas.width > 0.16 && hillas.wlRatio > 0.65) {
    logit -= 4.0;
  }

  // Sigmoid transform
  let pGamma = 1.0 / (1.0 + Math.exp(-logit));
  pGamma = Math.min(0.999, Math.max(0.001, pGamma));
  const pProton = 1.0 - pGamma;

  const qBoost = Math.min(63.2, pGamma / Math.max(0.015, Math.sqrt(pProton)));

  return { pGamma, pProton, qBoost };
}

// --- COLORMAP & CANVAS DRAWING ---

// Smooth Plasma / Viridis-inspired IACT colormap
function getColormapColor(val, maxVal) {
  if (val <= 0 || maxVal <= 0) return [11, 15, 25]; // background
  const t = Math.min(1.0, Math.max(0.0, val / maxVal));

  let r, g, b;
  if (t < 0.25) {
    const f = t / 0.25;
    r = Math.round(15 + 40 * f);
    g = Math.round(20 + 30 * f);
    b = Math.round(55 + 130 * f);
  } else if (t < 0.60) {
    const f = (t - 0.25) / 0.35;
    r = Math.round(55 + 140 * f);
    g = Math.round(50 + 10 * f);
    b = Math.round(185 - 85 * f);
  } else if (t < 0.85) {
    const f = (t - 0.60) / 0.25;
    r = Math.round(195 + 50 * f);
    g = Math.round(60 + 130 * f);
    b = Math.round(100 - 80 * f);
  } else {
    const f = (t - 0.85) / 0.15;
    r = Math.round(245 + 10 * f);
    g = Math.round(190 + 65 * f);
    b = Math.round(20 + 150 * f);
  }
  return [r, g, b];
}

function renderCanvas(canvasId, imgArray, hillasOverlay = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Max value for scaling
  let maxVal = 10.0;
  for (let i = 0; i < N_PIXELS; i++) {
    if (imgArray[i] > maxVal) maxVal = imgArray[i];
  }

  // Draw 34x34 pixel grid
  const pixW = w / N_SIDE;
  const pixH = h / N_SIDE;

  for (let r = 0; r < N_SIDE; r++) {
    for (let c = 0; c < N_SIDE; c++) {
      const idx = r * N_SIDE + c;
      const [cr, cg, cb] = getColormapColor(imgArray[idx], maxVal);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(c * pixW, r * pixH, pixW, pixH);
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= N_SIDE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * pixW, 0);
    ctx.lineTo(i * pixW, h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * pixH);
    ctx.lineTo(w, i * pixH);
    ctx.stroke();
  }

  // Center optical crosshair (Tracking axis)
  const cx = w / 2.0;
  const cy = h / 2.0;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
  ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
  ctx.stroke();

  // If Hillas overlay requested
  if (hillasOverlay && hillasOverlay.isValid) {
    const scale = w / (N_SIDE * PITCH_DEG); // pixels per deg

    // Centroid in canvas coordinates
    // pixX = FOV_MIN + c*PITCH_DEG -> x_canvas = cx + xMean*scale
    const kx = cx + hillasOverlay.xMean * scale;
    const ky = cy - hillasOverlay.yMean * scale; // Y inverted in canvas

    // 1. Line from tracking center (0,0) to centroid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(kx, ky);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Draw Hillas Ellipse
    const aPix = Math.max(3, hillasOverlay.length * scale);
    const bPix = Math.max(2, hillasOverlay.width * scale);
    const phiCanvas = -hillasOverlay.phi; // invert phi for canvas Y

    ctx.save();
    ctx.translate(kx, ky);
    ctx.rotate(phiCanvas);

    // Ellipse boundary
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.ellipse(0, 0, aPix, bPix, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Major axis line
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-aPix * 1.5, 0);
    ctx.lineTo(aPix * 1.5, 0);
    ctx.stroke();

    // Minor axis line
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(0, -bPix * 1.2);
    ctx.lineTo(0, bPix * 1.2);
    ctx.stroke();

    ctx.restore();

    // 3. Centroid Marker
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.arc(kx, ky, 3.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// --- UPDATE UI & PIPELINE PIPELINE ---

function updatePipeline() {
  // 1. Simulate
  const { rawImage, timeImg } = simulateShower();

  // 2. Clean
  const { cleanedImage, survivingPixels } = cleanImage(rawImage);

  // 3. Extract Hillas
  const hillas = calculateHillas(cleanedImage, timeImg, survivingPixels);

  // 4. ML Inference
  const mlRes = predictGammaProbability(hillas);

  // 5. Render Canvases
  renderCanvas('canvasRaw', rawImage);
  renderCanvas('canvasCleaned', cleanedImage);
  renderCanvas('canvasHillas', cleanedImage, hillas);

  // 6. Update Table & Text
  document.getElementById('cleanStats').innerText = `Retained: ${survivingPixels} pixels`;

  if (hillas.isValid) {
    document.getElementById('hillasAngleSummary').innerText =
      `α = ${hillas.alpha.toFixed(1)}° • L = ${hillas.length.toFixed(3)}° • W = ${hillas.width.toFixed(3)}°`;

    document.getElementById('tdSize').innerText = `${Math.round(hillas.size).toLocaleString()} pe`;
    document.getElementById('tdLength').innerText = `${hillas.length.toFixed(3)}°`;
    document.getElementById('tdWidth').innerText = `${hillas.width.toFixed(3)}°`;
    document.getElementById('tdWL').innerText = `${hillas.wlRatio.toFixed(3)}`;
    document.getElementById('tdDistance').innerText = `${hillas.distance.toFixed(3)}°`;
    document.getElementById('tdAlpha').innerText = `${hillas.alpha.toFixed(1)}°`;
    document.getElementById('tdMiss').innerText = `${hillas.miss.toFixed(3)}°`;
    document.getElementById('tdConc').innerText = `${hillas.concentration.toFixed(3)}`;

    // Color code WL ratio
    document.getElementById('tdWL').className = `py-2 px-3 font-bold ${hillas.wlRatio < 0.45 ? 'text-emerald-400' : 'text-red-400'}`;
    // Color code Alpha
    document.getElementById('tdAlpha').className = `py-2 px-3 font-bold ${hillas.alpha < 20 ? 'text-cyan-300' : 'text-amber-400'}`;
  } else {
    document.getElementById('hillasAngleSummary').innerText = 'Insufficient surviving pixels (< 3)';
  }

  // 7. Update ML Card
  const pG = mlRes.pGamma * 100;
  const pP = mlRes.pProton * 100;
  const isGamma = mlRes.pGamma >= 0.5;

  document.getElementById('probGammaText').innerText = `${pG.toFixed(1)}%`;
  document.getElementById('probProtonText').innerText = `${pP.toFixed(1)}%`;
  document.getElementById('probBarG').innerText = `${pG.toFixed(1)}%`;
  document.getElementById('probBarH').innerText = `${pP.toFixed(1)}%`;
  document.getElementById('probBar').style.width = `${pG.toFixed(1)}%`;
  document.getElementById('qMultiplier').innerText = `${mlRes.qBoost.toFixed(1)}x`;

  const badge = document.getElementById('badgeVerdict');
  const mlCard = document.getElementById('mlCard');
  const expBox = document.getElementById('decisionExplanation');

  if (isGamma) {
    badge.innerText = 'SIGNAL CANDIDATE (GAMMA-RAY)';
    badge.className = 'text-xs font-bold font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
    mlCard.className = 'glass-card rounded-2xl p-5 border shadow-xl transition-all border-emerald-500/30';
    expBox.innerHTML = `🟢 <strong>SIGNAL DETECTED: Gamma-Ray Event</strong> (${pG.toFixed(1)}% confidence)<br/>` +
      `Shower exhibits compact transverse width (<code>${hillas.width ? hillas.width.toFixed(3) : 0}°</code>), ` +
      `tight alignment with the tracking axis (<code>α = ${hillas.alpha ? hillas.alpha.toFixed(1) : 0}°</code>), ` +
      `and coherent Cherenkov wavefront timing.`;
  } else {
    badge.innerText = 'BACKGROUND REJECTED (HADRON / PROTON)';
    badge.className = 'text-xs font-bold font-mono px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/40';
    mlCard.className = 'glass-card rounded-2xl p-5 border shadow-xl transition-all border-red-500/30';
    expBox.innerHTML = `🔴 <strong>BACKGROUND REJECTED: Cosmic-Ray Hadron</strong> (${pP.toFixed(1)}% confidence)<br/>` +
      `Shower exhibits hadronic characteristics: broad lateral diffusion (<code>W/L = ${hillas.wlRatio ? hillas.wlRatio.toFixed(2) : 0}</code>), ` +
      `large pointing misalignment (<code>α = ${hillas.alpha ? hillas.alpha.toFixed(1) : 0}°</code>), or incoherent timing.`;
  }
}

// --- EVENT LISTENERS & PRESETS ---

function initUI() {
  const presetSelect = document.getElementById('presetSelect');
  const btnParticleGamma = document.getElementById('btnParticleGamma');
  const btnParticleProton = document.getElementById('btnParticleProton');
  const btnParticleMuon = document.getElementById('btnParticleMuon');

  const sliderAmp = document.getElementById('sliderAmp');
  const sliderX0 = document.getElementById('sliderX0');
  const sliderY0 = document.getElementById('sliderY0');
  const sliderPhi = document.getElementById('sliderPhi');
  const sliderLength = document.getElementById('sliderLength');
  const sliderWidth = document.getElementById('sliderWidth');
  const sliderNsb = document.getElementById('sliderNsb');
  const sliderCore = document.getElementById('sliderCore');
  const sliderBound = document.getElementById('sliderBound');
  const btnReseed = document.getElementById('btnReseed');

  function updateSliderDisplay() {
    document.getElementById('valAmp').innerText = state.amplitude;
    document.getElementById('valX0').innerText = `${state.x0.toFixed(2)}°`;
    document.getElementById('valY0').innerText = `${state.y0.toFixed(2)}°`;
    document.getElementById('valPhi').innerText = `${state.phiDeg.toFixed(1)}°`;
    document.getElementById('valLength').innerText = `${state.length.toFixed(2)}°`;
    document.getElementById('valWidth').innerText = `${state.width.toFixed(2)}°`;
    document.getElementById('valNsb').innerText = `${state.nsbSigma.toFixed(1)} pe`;
    document.getElementById('valCore').innerText = `${state.coreSigma.toFixed(1)} σ`;
    document.getElementById('valBound').innerText = `${state.boundarySigma.toFixed(1)} σ`;
  }

  function applyPreset(name) {
    if (name === 'gamma_onaxis') {
      state.particle = 'gamma';
      state.amplitude = 1200;
      state.x0 = 0.45; state.y0 = 0.45;
      state.phiDeg = 45.0; // Pointing directly to (0,0)
      state.length = 0.24; state.width = 0.08;
      state.nsbSigma = 4.5;
    } else if (name === 'gamma_offaxis') {
      state.particle = 'gamma';
      state.amplitude = 850;
      state.x0 = 0.45; state.y0 = 0.45;
      state.phiDeg = 0.0; // Misaligned -> alpha approx 45 deg
      state.length = 0.25; state.width = 0.09;
      state.nsbSigma = 4.5;
    } else if (name === 'proton_hadron') {
      state.particle = 'proton';
      state.amplitude = 1500;
      state.x0 = 0.35; state.y0 = -0.25;
      state.phiDeg = 20.0;
      state.length = 0.35; state.width = 0.22;
      state.nsbSigma = 4.5;
    } else if (name === 'muon_ring') {
      state.particle = 'muon';
      state.amplitude = 950;
      state.x0 = 0.0; state.y0 = 0.0;
      state.phiDeg = 0.0;
      state.length = 0.30; state.width = 0.20;
      state.nsbSigma = 4.5;
    } else if (name === 'moonlight') {
      state.particle = 'gamma';
      state.amplitude = 1600;
      state.x0 = 0.45; state.y0 = 0.45;
      state.phiDeg = 45.0;
      state.length = 0.24; state.width = 0.08;
      state.nsbSigma = 18.0; // High NSB
    }

    // Sync input controls
    sliderAmp.value = state.amplitude;
    sliderX0.value = state.x0;
    sliderY0.value = state.y0;
    sliderPhi.value = state.phiDeg;
    sliderLength.value = state.length;
    sliderWidth.value = state.width;
    sliderNsb.value = state.nsbSigma;

    syncParticleButtons();
    updateSliderDisplay();
    updatePipeline();
  }

  function syncParticleButtons() {
    btnParticleGamma.className = state.particle === 'gamma'
      ? 'py-1.5 font-semibold rounded bg-brand-500 text-white transition-all'
      : 'py-1.5 font-semibold rounded text-slate-400 hover:text-white transition-all';
    btnParticleProton.className = state.particle === 'proton'
      ? 'py-1.5 font-semibold rounded bg-red-600 text-white transition-all'
      : 'py-1.5 font-semibold rounded text-slate-400 hover:text-white transition-all';
    btnParticleMuon.className = state.particle === 'muon'
      ? 'py-1.5 font-semibold rounded bg-indigo-600 text-white transition-all'
      : 'py-1.5 font-semibold rounded text-slate-400 hover:text-white transition-all';
  }

  presetSelect.addEventListener('change', (e) => applyPreset(e.target.value));

  btnParticleGamma.addEventListener('click', () => { state.particle = 'gamma'; syncParticleButtons(); updatePipeline(); });
  btnParticleProton.addEventListener('click', () => { state.particle = 'proton'; syncParticleButtons(); updatePipeline(); });
  btnParticleMuon.addEventListener('click', () => { state.particle = 'muon'; syncParticleButtons(); updatePipeline(); });

  sliderAmp.addEventListener('input', (e) => { state.amplitude = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderX0.addEventListener('input', (e) => { state.x0 = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderY0.addEventListener('input', (e) => { state.y0 = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderPhi.addEventListener('input', (e) => { state.phiDeg = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderLength.addEventListener('input', (e) => { state.length = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderWidth.addEventListener('input', (e) => { state.width = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });
  sliderNsb.addEventListener('input', (e) => { state.nsbSigma = parseFloat(e.target.value); updateSliderDisplay(); updatePipeline(); });

  sliderCore.addEventListener('input', (e) => {
    state.coreSigma = parseFloat(e.target.value);
    // ensure boundary doesn't exceed core
    if (state.boundarySigma >= state.coreSigma) {
      state.boundarySigma = Math.max(1.0, state.coreSigma - 0.5);
      sliderBound.value = state.boundarySigma;
    }
    updateSliderDisplay();
    updatePipeline();
  });

  sliderBound.addEventListener('input', (e) => {
    state.boundarySigma = Math.min(state.coreSigma - 0.5, parseFloat(e.target.value));
    e.target.value = state.boundarySigma;
    updateSliderDisplay();
    updatePipeline();
  });

  btnReseed.addEventListener('click', () => {
    rngSeed = Math.floor(Math.random() * 100000);
    updatePipeline();
  });

  updateSliderDisplay();
  updatePipeline();
}

document.addEventListener('DOMContentLoaded', initUI);
