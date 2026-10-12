# CherenkovLens 🔭

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-black?logo=vercel&logoColor=white)](https://cherenkovlens.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-10tan%2Fcherenkovlens-blue?logo=github)](https://github.com/10tan/cherenkovlens)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-pep8-000000.svg)]()

> **Interactive Web Application**: Experience the live Cherenkov camera simulator, topological tail-cuts cleaning, analytical Hillas eigensolver, and real-time machine learning inference directly in your browser at **[https://cherenkovlens.vercel.app](https://cherenkovlens.vercel.app)**.

**CherenkovLens** is an end-to-end, pure-Python simulation, image cleaning, feature extraction, and machine learning classification pipeline for **Imaging Atmospheric Cherenkov Telescopes (IACTs)**.

The architecture is modeled on the data reduction and analysis chains developed for high-altitude gamma-ray observatories in the Himalayas: specifically the **Major Atmospheric Cerenkov Experiment (MACE)** telescope at Hanle, Ladakh (altitude $\sim 4,270\text{ m}$ a.s.l., mirror diameter $21\text{ m}$) and the **High Altitude GAmma Ray (HAGAR)** wavefront sampling array.

---

## 🌐 Live Interactive Deployments

| Platform | Access Link | Description |
| :--- | :--- | :--- |
| **Vercel Web App** | **[cherenkovlens.vercel.app](https://cherenkovlens.vercel.app)** | Instant client-side interactive 34×34 MACE camera simulator, real-time tail-cuts cleaning sliders, Hillas eigensolver overlay, dual-class ML probabilities, and KaTeX mathematical reference. |
| **Local Streamlit Studio** | `streamlit run app/streamlit_app.py` | Local Python analysis laboratory featuring interactive preset scenarios, tri-panel event visualizer, diagnostic benchmark tables, and live Random Forest pipeline inference. |
| **GitHub Repository** | **[github.com/10tan/cherenkovlens](https://github.com/10tan/cherenkovlens)** | Complete open-source codebase, automated pytest test suite, benchmark datasets, trained scikit-learn models, and publication figures. |

---

## 🌌 Physical Motivation & Theoretical Background

### 1. Atmospheric Cherenkov Emission Microphysics
When ultra-high-energy cosmic gamma rays or cosmic-ray nuclei (primarily protons, helium) enter Earth's upper atmosphere ($\sim 20\text{--}30\text{ km}$ altitude), they trigger cascading relativistic particle showers (Extensive Air Showers, EAS). As secondary relativistic electrons and positrons travel faster than the local phase speed of light in the dielectric atmospheric medium:

$$v > \frac{c}{n(h)} \quad \iff \quad \beta n(h) > 1$$

they polarize adjacent air molecules, producing an asymmetric dipole wake that emits a brief ($3\text{--}10\text{ ns}$), faint blue/near-UV flash of **Cherenkov radiation**.

The atmospheric refractive index $n(h)$ varies with altitude $h$ following the barometric density profile:

$$n(h) = 1 + \eta_0 \exp\left(-\frac{h}{h_0}\right)$$

where $\eta_0 \approx 2.9 \times 10^{-4}$ is the sea-level refractivity and $h_0 \approx 7.1\text{ km}$ is the atmospheric density scale height.

Cherenkov radiation is beamed along a forward conical wavefront with half-opening angle $\theta_C(h)$:

$$\cos\theta_C(h) = \frac{1}{\beta n(h)} \quad \implies \quad \theta_C(h) \approx \sqrt{2(n(h) - 1)} \approx 1.0^\circ \text{ to } 1.4^\circ$$

The spectral photon emission density per unit path length $dx$ is governed by the classical **Frank-Tamm equation**:

$$\frac{d^2 N_\gamma}{dx \, d\lambda} = \frac{2\pi \alpha}{\lambda^2} \left(1 - \frac{1}{\beta^2 n^2(\lambda)}\right) = \frac{2\pi \alpha}{\lambda^2} \sin^2\theta_C$$

Because the photon production rate scales as $dN/d\lambda \propto \lambda^{-2}$, Cherenkov light is intensely concentrated in the near-UV and optical blue wavelengths ($\lambda \approx 300\text{--}450\text{ nm}$). On the ground, this conical emission forms an optical light pool spanning a characteristic radius $R_C \approx h \tan\theta_C \sim 100\text{--}130\text{ m}$ around the shower core.

---

### 2. Shower Cascade Dynamics: Electromagnetic vs. Hadronic Showers

IACT cameras record distinct morphological signatures driven by fundamental particle interaction physics:

```
            PRIMARY GAMMA RAY (γ)                      PRIMARY PROTON (p)
                     │                                         │
       Pair Production (γ → e⁺e⁻)                   Inelastic QCD Collision
                     │                                         │
              Bremsstrahlung                       Multi-Core Hadronic Tree
          (e± → e± + γ Brems)                     (p + Air → π±, π⁰, K±, p, n)
                     │                                   ┌─────┴─────┐
            Compact, Smooth, Elongated                   │           │
                 Elliptical Pool                   π⁰ → 2γ      π± → μ± + ν
          (Coulomb Multiple Scattering)         Displaced Sub-cascades  Muon Rings
```

1. **Electromagnetic Cascades ($\gamma$-ray Primaries)**:
   - Governed strictly by Quantum Electrodynamics (QED): alternating cycles of pair production ($\gamma + Z \to e^+ e^- + Z$) and bremsstrahlung ($e^\pm + Z \to e^\pm \gamma + Z$).
   - Longitudinal development is set by the radiation length in air, $X_0 \approx 36.66\text{ g/cm}^2$.
   - Transverse lateral spread is constrained purely by Coulomb multiple scattering within the Molière radius $R_M \approx 9.3\text{ g/cm}^2 / \rho(h)$.
   - **Camera Morphology**: Smooth, compact, highly regular 2D ellipses with tight temporal coherence ($\sigma_{\text{jitter}} \lesssim 0.7\text{--}0.9\text{ ns}$).

2. **Hadronic Cascades (Proton / Nuclei Primaries)**:
   - Governed by Quantum Chromodynamics (QCD) inelastic collisions with atmospheric nitrogen and oxygen nuclei ($p + \text{Air} \to \pi^\pm, \pi^0, K^\pm, p, n$).
   - Nuclear interaction length $\lambda_I \approx 84.9\text{ g/cm}^2 \gg X_0$.
   - Secondary pions acquire significant transverse momentum ($p_T \sim 350\text{ MeV}/c$), causing sub-showers to bifurcate widely from the primary trajectory.
   - Neutral pions ($\pi^0$) decay quasi-instantaneously ($\tau \sim 8.5 \times 10^{-17}\text{ s}$) into twin gamma rays ($\pi^0 \to 2\gamma$), spawning displaced electromagnetic sub-cascades, while charged pions ($\pi^\pm$) decay into penetrating muons ($\mu^\pm$).
   - **Camera Morphology**: Clumpy, irregular, multi-blob images with broad transverse extent, isolated pixel islands, and substantial arrival-time dispersion ($\sigma_{\text{jitter}} \sim 1.3\text{--}2.5\text{ ns}$).

---

### 3. The High-Altitude Hanle Advantage (MACE Observatory)
The **Major Atmospheric Cerenkov Experiment (MACE)** telescope operates at Hanle in the Changthang Plateau, Ladakh ($32^\circ 46' 46''\,\text{N}, 78^\circ 58' 35''\,\text{E}$, altitude $\sim 4,270\text{ m}$ a.s.l.):

1. **Reduced Atmospheric Overburden**: The atmospheric depth at Hanle is only $\approx 600\text{ g/cm}^2$, compared to $\approx 1,030\text{ g/cm}^2$ at sea level.
2. **Shower Maximum Proximity**: For primary energies in the range $20\text{--}100\text{ GeV}$, extensive air showers reach their maximum particle number ($X_{\max}$) at an atmospheric depth of $450\text{--}650\text{ g/cm}^2$. Because the MACE reflector sits directly inside this altitude window, Cherenkov photon densities are up to **4 to 5 times higher** than at sea level.
3. **Ultra-Low Energy Threshold**: The combination of high altitude and a $21\text{ m}$ diameter quasi-parabolic tessellated reflector ($356\text{ m}^2$ collecting area) lowers MACE's energy threshold down to $\approx 20\text{--}30\text{ GeV}$, bridging the gap between space telescopes (Fermi-LAT) and ground-based arrays (H.E.S.S., MAGIC, VERITAS).

---

## 🛠️ End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph SIM["1. Atmospheric Shower Simulation"]
        A1["Gamma Showers: Rotated 2D Elliptical Gaussian"] --> S1["Signal Image S_i (pe)"]
        A2["Proton Showers: Multi-Blob QCD Superposition"] --> S1
        A3["Muon Rings: Annular Geometric Arc"] --> S1
        A4["Wavefront Propagation: Linear Slope + PMT Jitter"] --> T1["Arrival-Time Image t_i (ns)"]
        N1["Night Sky Background (NSB) Noise Model"] --> S2["Observed Frame: q_i = Poisson(S_i) + N(0, σ_nsb²)"]
    end

    subgraph CLN["2. Topological Image Cleaning"]
        S2 --> C1{"Cleaning Algorithm"}
        C1 -->|"Core Cut: q_i ≥ 6.0 σ_nsb<br/>Boundary Cut: q_i ≥ 3.0 σ_nsb<br/>8-Connected BFS Island Labeling"| C2["Two-Level Tail-Cuts Cleaned Frame"]
        C1 -->|"Single Hard Cut: q_i ≥ 4.5 σ_nsb"| C3["Naive Cleaned Frame (Baseline)"]
    end

    subgraph HIL["3. Analytical Moment Extraction"]
        C2 --> H1["Analytical Closed-Form 2x2 Eigensolver"]
        H1 --> H2["Hillas Spatial Parameters: S, L, W, W/L, D, α, M, conc"]
        T1 --> H3["Timing Moments: Gradient ∇_t, Residual RMS σ_t"]
        H2 & H3 --> F1["Vetted 11-Dimensional Feature Vector"]
    end

    subgraph ML["4. Machine Learning & Q-Factor Optimization"]
        F1 --> M1["Stratified 80/20 Train/Test Split (7,992 Test Events)"]
        M1 --> M2["Random Forest (140 Trees, Depth 12)"]
        M1 --> M3["Histogram Gradient Boosting"]
        M1 --> M4["Standardized Logistic Regression"]
        M2 & M3 & M4 --> Q1["Telescope Quality Factor Q(θ) = ε_γ(θ) / √ε_p(θ)"]
        Q1 --> Q2["Optimal Selection Cut θ* (Peak Q = 59.36)"]
    end

    subgraph OUT["5. Deployment & Interactive Inspection"]
        M2 --> APP1["Streamlit Studio (app/streamlit_app.py)"]
        F1 --> APP2["Vercel Web App (https://cherenkovlens.vercel.app)"]
        Q2 --> FIG["8 High-Resolution Publication Figures (300 DPI)"]
    end
```

---

## 📐 Mathematical Derivations

### 1. Two-Level Tail-Cuts Topological Cleaning Formulation
Night Sky Background (NSB) photons produce independent Gaussian/Poisson fluctuations across all $1,156$ camera pixels. A simple hard amplitude threshold faces a severe dilemma: lowering the threshold introduces spurious noise pixels that distort second moments, while raising it discards the faint lateral fringes of real showers.

The two-level tail-cuts algorithm addresses this through topological adjacency:
1. **Core Condition**: A pixel $i$ is designated as a core pixel if:
   $$q_i \ge q_{\text{core}} = k_c \cdot \sigma_{\text{nsb}} \quad (k_c = 6.0)$$
2. **Boundary Condition**: A pixel $j$ is designated as a boundary pixel if:
   $$q_j \ge q_{\text{bound}} = k_b \cdot \sigma_{\text{nsb}} \quad (k_b = 3.0) \quad \text{and} \quad \exists \, i \in \mathcal{N}_8(j) \text{ such that } q_i \ge q_{\text{core}}$$
   where $\mathcal{N}_8(j)$ represents the 8-connected Moore neighborhood on the square grid (or 6-connected hexagonal ring).
3. **Graph Island Connected-Component Filtering**: The retained pixel set forms an adjacency graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. We compute connected components via breadth-first search (BFS). Any isolated island containing fewer than $N_{\min} = 2$ pixels or lacking at least one core pixel is discarded.

---

### 2. Analytical Closed-Form $2 \times 2$ Covariance Eigensolver
Let $q_i$ denote the cleaned photo-electron charge in pixel $i$ at angular camera coordinates $(x_i, y_i)$.

#### Zeroth and First Moments:
The total shower size (intensity zero-moment) and center of gravity (centroid) are:

$$S = \sum_{i \in \text{clean}} q_i, \quad \langle x \rangle = \frac{1}{S}\sum_{i \in \text{clean}} q_i x_i, \quad \langle y \rangle = \frac{1}{S}\sum_{i \in \text{clean}} q_i y_i$$

#### Central Second Moments (Covariance Tensor):
Let $dx_i = x_i - \langle x \rangle$ and $dy_i = y_i - \langle y \rangle$. The components of the spatial covariance matrix $\mathbf{\Sigma}$ are:

$$\sigma_{xx} = \frac{1}{S}\sum_{i} q_i dx_i^2, \quad \sigma_{yy} = \frac{1}{S}\sum_{i} q_i dy_i^2, \quad \sigma_{xy} = \frac{1}{S}\sum_{i} q_i dx_i dy_i$$

#### Exact Algebraic Eigensolution:
The eigenvalues of $\mathbf{\Sigma} = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{pmatrix}$ satisfy the characteristic equation $\det(\mathbf{\Sigma} - \lambda \mathbf{I}) = 0$:

$$\lambda^2 - (\sigma_{xx} + \sigma_{yy})\lambda + (\sigma_{xx}\sigma_{yy} - \sigma_{xy}^2) = 0$$

Defining coordinate variance asymmetry $\Delta \equiv \sigma_{xx} - \sigma_{yy}$ and discriminant $d \equiv \sqrt{\Delta^2 + 4\sigma_{xy}^2}$, the closed-form eigenvalues $\lambda_\pm$ are:

$$\lambda_+ = \frac{\sigma_{xx} + \sigma_{yy} + d}{2}, \quad \lambda_- = \frac{\sigma_{xx} + \sigma_{yy} - d}{2}$$

The physical **Hillas Length** ($L$) and **Width** ($W$) are defined as the root-mean-square spreads along the principal axes:

$$\text{Length} = \sqrt{\lambda_+} = \sqrt{\frac{\sigma_{xx} + \sigma_{yy} + d}{2}}$$

$$\text{Width} = \sqrt{\lambda_-} = \sqrt{\frac{\sigma_{xx} + \sigma_{yy} - d}{2}}$$

$$\text{Compactness Ratio} = \frac{\text{Width}}{\text{Length}} = \sqrt{\frac{\sigma_{xx} + \sigma_{yy} - d}{\sigma_{xx} + \sigma_{yy} + d}}$$

#### Shower Azimuth / Orientation Angle ($\phi$):
The major axis eigenvector aligns with azimuth angle $\phi \in [-\pi/2, \pi/2]$:

$$\tan(2\phi) = \frac{2\sigma_{xy}}{\sigma_{xx} - \sigma_{yy}} \implies \phi = \frac{1}{2} \arctan2(2\sigma_{xy}, \sigma_{xx} - \sigma_{yy})$$

#### Geometrical Pointing Parameters:
1. **Distance ($D$)**: Radial displacement from the camera tracking center $(0, 0)$ to shower centroid:
   $$D = \sqrt{\langle x \rangle^2 + \langle y \rangle^2}$$
2. **Alpha Angle ($\alpha$)**: The angle between the shower major axis unit vector $\vec{u} = (\cos\phi, \sin\phi)$ and the centroid radius vector $\vec{c} = (\langle x \rangle, \langle y \rangle)$:
   $$\cos\alpha = \frac{|\vec{u} \cdot \vec{c}|}{\|\vec{u}\| \|\vec{c}\|} = \frac{|\langle x \rangle \cos\phi + \langle y \rangle \sin\phi|}{D} \implies \alpha = \arccos(\cos\alpha) \in [0^\circ, 90^\circ]$$
   - For an **on-axis point source**, the shower axis points directly back to the origin, yielding $\alpha \approx 0^\circ$ (peaked within $\lesssim 10^\circ$).
   - For **isotropic cosmic-ray hadrons**, primary trajectories are distributed uniformly across the celestial sphere, yielding a flat probability density:
     $$P_{\text{hadron}}(\alpha) = \frac{1}{90^\circ} \quad \forall \, \alpha \in [0^\circ, 90^\circ]$$
3. **Miss Parameter ($M$)**: Perpendicular distance from camera center to the major axis line:
   $$M = D \sin\alpha = |\langle x \rangle \sin\phi - \langle y \rangle \cos\phi|$$
4. **Concentration ($\text{conc}$)**: Fraction of total shower light in the two brightest core pixels:
   $$\text{conc} = \frac{q_{(1)} + q_{(2)}}{S}$$

#### Arrival-Time Gradient and RMS:
The coordinate along the major axis is $p_i = dx_i \cos\phi + dy_i \sin\phi$. Fitting arrival times $t_i = t_0 + \nabla_t \cdot p_i$ via weighted linear regression yields the linear propagation slope $\nabla_t$ and the residual timing fluctuation $\sigma_t$ (incoherent sub-cascade spread):

$$\nabla_t = \frac{\sum q_i p_i (t_i - \bar{t})}{\sum q_i p_i^2}, \quad \sigma_t = \sqrt{\frac{1}{S}\sum q_i (t_i - (t_0 + \nabla_t p_i))^2}$$

---

### 3. Telescope Quality Factor ($Q$) Optimization & Observing Time
In observational gamma-ray astronomy, background hadronic cosmic rays outnumber gamma rays by $10^3\text{--}10^5 : 1$. Under Poissonian statistics (Li & Ma, 1983), the pre-cut detection significance $\mathcal{S}_0$ is:

$$\mathcal{S}_0 \approx \frac{N_{\text{excess}}}{\sqrt{N_{\text{background}}}} = \frac{N_\gamma}{\sqrt{N_p}}$$

Applying a machine learning decision cut $\theta \in [0, 1]$ retains fraction $\epsilon_\gamma(\theta)$ of true gamma rays (True Positive Rate) and fraction $\epsilon_p(\theta)$ of background hadrons (False Positive Rate):

$$\mathcal{S}(\theta) \approx \frac{\epsilon_\gamma(\theta) N_\gamma}{\sqrt{\epsilon_p(\theta) N_p}} = \left( \frac{\epsilon_\gamma(\theta)}{\sqrt{\epsilon_p(\theta)}} \right) \cdot \frac{N_\gamma}{\sqrt{N_p}} = Q(\theta) \cdot \mathcal{S}_0$$

The **Telescope Quality Factor** $Q(\theta)$ is defined as:

$$Q(\theta) \equiv \frac{\epsilon_\gamma(\theta)}{\sqrt{\epsilon_p(\theta)}}$$

#### Inverse-Square Observation Time Scaling:
Because detection significance accumulates with exposure time as $\mathcal{S} \propto \sqrt{T_{\text{obs}}}$, the required telescope tracking time to attain a canonical $5\sigma$ discovery significance scales inversely with $Q^2$:

$$T_{\text{required}}(\theta) = \frac{T_0}{Q^2(\theta)}$$

| Classifier Cut State | Quality Factor $Q$ | Required Telescope Observing Time |
| :--- | :---: | :---: |
| **No Classification Cut (Raw)** | $Q = 1.00$ | $100.0\text{ hours}$ |
| **Simple Hillas Cut (Historical)** | $Q \approx 2.50$ | $16.0\text{ hours}$ |
| **Gradient Boosting ($\theta^* = 0.98$)** | $Q = 30.82$ | $6.3\text{ minutes}$ |
| **Random Forest ($\theta^* = 0.98$)** | **$Q = 59.36$** | **$1.7\text{ minutes}$** |

An observation that would otherwise require **100 hours** of telescope time achieves the identical discovery significance in **under 2 minutes** using the trained Random Forest classifier!

---

## 📊 Empirical Results & Validation Benchmarks

> **Empirical Integrity**: All figures and tables below reflect actual numerical outputs produced by `run_all.sh` across 40,000 synthetic events (master seed `42`).

### 1. Image Cleaning: Tail-Cuts ($6.0 / 3.0\sigma$) vs. Naive ($4.5\sigma$)
Evaluated across 2,000 simulated events under nominal $\sigma_{\text{nsb}} = 4.5\text{ pe/pixel}$:

| Cleaning Method | Signal Pixel Retention | Noise Pixel Leakage / Event | Shower Size Reconstruction Bias |
| :--- | :---: | :---: | :---: |
| **Naive Single-Cut ($4.5\sigma$)** | 54.12% | 0.0065 pix | -7.11% |
| **Two-Level Tail-Cuts ($6.0 / 3.0\sigma$)** | **62.23%** | 0.0285 pix | **-4.24%** |
| **Gain / Advantage** | **+8.11 percentage points** | Minimal leakage | **+2.87% bias improvement** |

<p align="center">
  <img src="figures/cleaning_study.png" width="98%" alt="Image Cleaning Comparison" />
</p>
*Figure 1: Two-level tail-cuts topological island reconstruction versus naive single-cut cleaning, demonstrating signal pixel recovery (+8.11%) and reduced shower Size reconstruction bias.*

---

### 2. Hillas Parameter Ground-Truth Validation
Reconstructed parameters were compared against analytical ground-truth values for 1,000 clean, noise-free gamma showers:

| Hillas Dimension | Pearson Correlation ($r$) | Mean Relative Error |
| :--- | :---: | :---: |
| **Length ($L$)** | **0.9976** | **1.32%** |
| **Width ($W$)** | **0.9988** | **1.11%** |
| **Distance ($D$)** | **0.9997** | **< 0.50%** |
| **Alpha Angle ($\alpha$)** | **0.9963** | **< 1.0°** |

<p align="center">
  <img src="figures/hillas_validation.png" width="98%" alt="Hillas Parameter Validation" />
</p>
*Figure 2: Ground-truth validation of reconstructed Hillas parameters showing Pearson $r > 0.996$ across all physical shower dimensions.*

---

### 3. Machine Learning Gamma/Hadron Classification
Evaluated on a stratified test set of **7,992 events** (4,000 gamma rays, 3,992 protons) using strictly vetted morphological and timing features:

| Classifier Model | ROC AUC | Test Accuracy | F1-Score | Precision | Recall | Peak $Q$-Factor ($Q_{\max}$) | Optimal Threshold ($\theta^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9999** | **99.57%** | **0.9958** | **99.43%** | **99.73%** | **59.36** | **0.98** |
| **Histogram Gradient Boosting** | 0.9998 | 99.61% | 0.9961 | 99.43% | 99.80% | 30.82 | 0.98 |
| **Standardized Logistic Regression** | 0.9996 | 99.30% | 0.9930 | 99.20% | 99.40% | 24.15 | 0.94 |

<p align="center">
  <img src="figures/roc_curve.png" width="49%" alt="ROC Curves" />
  <img src="figures/q_factor_curve.png" width="49%" alt="Quality Factor Curve" />
</p>
*Figures 3 & 4: Left: Stratified test set ROC curves comparing Random Forest, Histogram Gradient Boosting, and Logistic Regression. Right: Quality Factor $Q(\theta)$ versus decision threshold $\theta$, showing peak $Q_{\max} = 59.36$ with $93.95\%$ gamma efficiency and $0.025\%$ proton survival.*

---

### 4. Morphological Parameter Separation & Feature Importances

| Feature Name | Feature Importance (Gini) | Physical Role |
| :--- | :---: | :--- |
| `time_rms` | **36.28%** | Wavefront coherence: electromagnetic (0.7 ns) vs hadronic turbulence (1.3 ns) |
| `width` | **18.34%** | Lateral spread: pure Coulomb scattering vs secondary hadronic sub-cascades |
| `length` | **13.87%** | Longitudinal cascade development in air |
| `alpha` | **12.69%** | Collinear pointing alignment with camera tracking axis |
| `miss` | **7.68%** | Perpendicular impact distance to optical center |
| `concentration`| **3.70%** | Core compactness: dense central core vs diffuse fragmented blobs |
| `time_gradient`| **2.05%** | Downward propagation velocity along shower major axis |
| `wl_ratio` | **1.98%** | Elongated ellipse compactness discriminator |
| `n_pixels` | **1.92%** | Size of connected island on the camera plane |
| `distance` | **1.22%** | Impact distance from camera center |
| `size` | **0.28%** | Total reconstructed photo-electron charge |

<p align="center">
  <img src="figures/alpha_distribution.png" width="32.5%" alt="Alpha Distribution" />
  <img src="figures/width_length_scatter.png" width="32.5%" alt="Width vs Length Scatter" />
  <img src="figures/feature_importance.png" width="32.5%" alt="Feature Importance" />
</p>
*Figures 5, 6 & 7: Left: Reconstructed Alpha angle distribution peaking at $\alpha \approx 0^\circ$ for on-axis gamma rays. Middle: Width vs Length morphological manifold separation. Right: Random Forest Gini feature importances.*

---

### 5. Night Sky Background (NSB) Noise Robustness Study
The complete reduction and classification pipeline was re-evaluated under elevated NSB photon noise levels ($1\times, 2\times, 4\times, 8\times$ nominal noise $\sigma_{\text{nsb}} = 4.5, 9.0, 18.0, 36.0\text{ pe/pixel}$), simulating observation conditions ranging from dark nights to bright moonlight:

| NSB Level | $\sigma_{\text{nsb}}$ (pe/pixel) | Tail-Cuts Surviving Events | Tail-Cuts ROC AUC | Naive ROC AUC | Tail-Cuts $Q_{\max}$ | Naive $Q_{\max}$ | Tail-Cuts Advantage ($\Delta Q$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1\times$ (Nominal)** | 4.5 pe | 2,997 | **0.9999** | 0.9997 | **17.26** | 17.21 | +0.06 |
| **$2\times$** | 9.0 pe | 2,927 | **0.9999** | 0.9994 | **16.88** | 16.29 | **+0.60** |
| **$4\times$** | 18.0 pe | 2,534 | **0.9982** | 0.9994 | 15.03 | 15.28 | -0.25 |
| **$8\times$ (Bright Moon)** | 36.0 pe | 2,017 | **0.9982** | 0.9904 | **9.71** | 7.89 | **+1.81** |

<p align="center">
  <img src="figures/noise_robustness.png" width="95%" alt="NSB Noise Robustness" />
</p>
*Figure 8: Classifier ROC AUC and Quality Factor across elevated NSB noise levels with and without tail-cuts topological cleaning.*

---

## 📁 Repository Structure

```text
cherenkovlens/
├── README.md                          # Comprehensive documentation, math & benchmarks
├── pyproject.toml                     # PEP 621 package metadata & dependencies
├── requirements.txt                   # Pinned dependency requirements
├── run_all.sh                         # Complete pipeline execution orchestrator
├── config/
│   └── default.yaml                   # Tunable camera, noise, cleaning, and model parameters
├── src/cherenkovlens/
│   ├── __init__.py                    # Package exports
│   ├── camera.py                      # Square & hexagonal camera geometries (MACE 34x34)
│   ├── simulate.py                    # Rotated 2D Gaussian, multi-blob, muon ring, and timing
│   ├── cleaning.py                    # Two-level tail-cuts (6/3 sigma BFS) & naive cleaning
│   ├── hillas.py                      # Analytical closed-form 2x2 eigensolver & Hillas moments
│   ├── dataset.py                     # Multiprocess chunked dataset generator
│   ├── classify.py                    # Scikit-learn pipelines, metrics, and Q-factor curve
│   ├── plotting.py                    # 8 publication-grade visualization routines (300 DPI)
│   └── utils.py                       # Config loader, reproducible seed manager, logging
├── scripts/
│   ├── generate_dataset.py            # Dataset simulation CLI (40,000 events)
│   ├── run_cleaning_study.py          # Tail-cuts vs naive cleaning benchmark CLI
│   ├── validate_hillas.py             # Eigensolver correctness validation CLI
│   ├── train_classifier.py            # Model training & metric serialization CLI
│   └── run_noise_study.py             # Multi-level NSB noise benchmark CLI
├── app/
│   └── streamlit_app.py               # Interactive Streamlit application with presets & diagnostics
├── web/
│   ├── index.html                     # Vercel single-page interactive web application
│   ├── app.js                         # Browser client-side simulation & Hillas eigensolver
│   ├── figures/                       # Publication figures served statically on Vercel
│   └── .vercel/project.json           # Vercel project deployment link
├── notebooks/
│   └── walkthrough.ipynb              # Narrated Jupyter walkthrough notebook
├── tests/
│   ├── test_simulate.py               # Camera geometry & shower simulation unit tests
│   ├── test_cleaning.py               # Tail-cuts topological island connectivity tests
│   ├── test_hillas.py                 # Analytical eigensolver vs numpy.linalg.eigh tests
│   └── test_classify.py               # Feature gating, pipeline execution & Q-factor tests
├── figures/                           # Saved publication-quality PNG figures (300 DPI)
├── models/                            # Serialized scikit-learn models (classifier_rf.joblib)
└── results/                           # Empirical metrics (JSON and CSV)
```

---

## 🚀 Installation & Local Execution

### 1. Clone & Setup Python Environment
```bash
git clone https://github.com/10tan/cherenkovlens.git
cd cherenkovlens

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e .
```

### 2. Run the Unit Test Suite
```bash
pytest tests/ -v
```
Output:
```text
============================== 15 passed in 7.29s ==============================
```

### 3. Run the Entire Pipeline End-to-End
```bash
chmod +x run_all.sh
./run_all.sh
```
This executes:
1. Dataset simulation (40,000 events with 11 Hillas + timing features).
2. Tail-cuts vs. naive image cleaning benchmark.
3. Hillas eigensolver ground-truth validation.
4. Model training (Random Forest, Histogram Gradient Boosting, Logistic Regression).
5. NSB noise robustness benchmark ($1\times, 2\times, 4\times, 8\times$ noise levels).
6. Generation of all 8 publication figures in `figures/`.

### 4. Launch the Interactive Streamlit Studio
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` to interactively explore shower profiles, compare cleaning algorithms, and observe real-time classification responses.

### 5. Deploy to Vercel
The web studio in `web/` deploys directly to Vercel:
```bash
npx vercel deploy web --prod --yes
```

---

## 🔭 Telescope Specifications (MACE & HAGAR)

| Observatory Parameter | MACE Telescope | HAGAR Array |
| :--- | :--- | :--- |
| **Site Location** | Hanle, Ladakh, India | Hanle, Ladakh, India |
| **Geographic Coordinates** | $32^\circ 46' 46''\,\text{N}, 78^\circ 58' 35''\,\text{E}$ | $32^\circ 46' 46''\,\text{N}, 78^\circ 58' 35''\,\text{E}$ |
| **Altitude a.s.l.** | **$4,270\text{ m}$** | **$4,270\text{ m}$** |
| **Atmospheric Overburden** | $\approx 600\text{ g/cm}^2$ | $\approx 600\text{ g/cm}^2$ |
| **Optical Reflector** | $21\text{ m}$ diameter quasi-parabolic ($356\text{ m}^2$) | 7 telescopes, each with 7 mirrors ($31\text{ m}^2$ total) |
| **Focal Length** | $25\text{ m}$ ($f/d \approx 1.19$) | $4.45\text{ m}$ |
| **Camera Configuration** | 1,088 photomultiplier tubes (PMTs) | 7 PMTs per telescope (wavefront sampling) |
| **Angular Pixel Pitch** | $0.1^\circ$ (inner 576 PMTs), $0.2^\circ$ (outer 512 PMTs) | Wavefront arrival timing array |
| **Field of View (FoV)** | $4.0^\circ \times 4.0^\circ$ | $3.0^\circ$ acceptance cone |
| **Energy Threshold** | **$\approx 20\text{--}30\text{ GeV}$** | $\approx 200\text{ GeV}$ |

---

## ⚖️ Limitations & Scope

1. **Parametric Shower Simulation**: Cherenkov shower images are synthesized using rotated 2D elliptical Gaussian profiles (for electromagnetic gammas) and multi-blob superposition (for hadronic protons) rather than full 3D Monte Carlo particle transport through an atmospheric density model (e.g., CORSIKA / CORSIKA 8).
2. **Simplified Camera Grid**: The camera model utilizes an analytical square grid ($34 \times 34 = 1,156$ pixels, pitch $0.1^\circ$). Physical MACE cameras feature graded hexagonal PMTs ($0.1^\circ$ inner, $0.2^\circ$ outer) with Winston light cones and individual channel high-voltage gains.
3. **Trigger Logic**: Tail-cuts cleaning is evaluated directly on digitized charges; hardware coincidence trigger simulation (e.g., majority coincidence of 3–4 adjacent pixels within 3–5 ns) is not explicitly modeled.

---

## 📜 References & Further Reading

1. **Hillas, A. M.** (1985). *Cerenkov light images of EAS produced by primary gamma rays and by nuclei*. Proceedings of the 19th International Cosmic Ray Conference (ICRC), La Jolla, 3, 445–448.
2. **Koul, R. et al.** (2011). *The MACE Gamma-Ray Telescope at Hanle, Ladakh*. Nuclear Instruments and Methods in Physics Research Section A, 692, 174–180.
3. **Chitnis, V. R. et al.** (2009). *The High Altitude GAmma Ray (HAGAR) Telescope Array at Hanle*. High Energy Gamma-Ray Astronomy, AIP Conference Proceedings, 1085, 747–750.
4. **Li, T.-P. & Ma, Y.-Q.** (1983). *Analysis methods for results in gamma-ray astronomy*. The Astrophysical Journal, 272, 317–324.
5. **Frank, I. M. & Tamm, I. E.** (1937). *Coherent visible radiation of fast electrons passing through matter*. Comptes Rendus de l'Académie des Sciences de l'URSS, 14(3), 109–114.
6. **Heitler, W.** (1954). *The Quantum Theory of Radiation* (3rd ed.). Oxford University Press.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
