# Absolute Spectral Calibration Procedure

Converts raw camera counts from the 80-fiber ELI-MAY side-scatter spectrometer into an absolute spectral energy density (J/nm) at the fiber entrance, using an independent power-meter measurement of the infrared calibration source as the absolute reference.

## 1. Pipeline summary

The calibration links two independent measurements — a **camera image** of the calibration source and **power-meter readings** at each fiber position — through a common physical quantity: the power spectral density incident on each fiber.

1. **Camera side.** Subtract background from the calibration image, integrate each of the 80 fiber traces vertically (95% of the inter-fiber spacing, with fractional-pixel weighting), and divide by exposure time to get a count rate $\dot C_{f,j}$ per fiber $f$ and wavelength bin $j$.
2. **Power-meter side.** Correct the power meter's displayed reading (taken at a fixed 750 nm setting) for the detector's actual spectral responsivity to the broadband source, then geometrically project that reading onto the fiber core (aperture-area ratio + inverse-square distance correction) to get the total broadband power entering each fiber.
3. **Spectral shape.** Take the source's true normalized spectral shape (measured independently with the fiber spectrometer), apply the long-pass filter model, and use it to distribute the fiber's total power into a known spectral power density $P_{\lambda,f,j}$ [W/nm].
4. **Calibration coefficient.** Divide the known spectral power density by the measured count rate: $K_{f,j} = P_{\lambda,f,j} / \dot C_{f,j}$, in J·nm⁻¹·count⁻¹. This is the inverse spectral response — not a transmission curve — computed independently per fiber and per wavelength bin.
5. **Cleanup.** Smooth $K_{f,j}$ (10 nm moving average in log space), replace abnormal fiber curves with the mean of the others, and force $K_{f,j}=0$ below 660 nm (below the long-pass cutoff, where no real signal is physically possible).
6. **Application.** For any shot, multiply the measured counts by $K_{f,j}$ to get calibrated spectral energy density.

The key conceptual point: the ND filter and every other unmodeled element of the optical chain (fiber transmission, spectrometer efficiency, camera QE) are **not** modeled explicitly — their combined attenuation is absorbed empirically into the measured count rate $\dot C_{f,j}$, because the calibration is built as a ratio of *known incident power* to *measured counts*, not as a product of individually characterized transmission curves.

## 2. Camera image processing

The calibration image (source on) and background image (source off) are subtracted pixel-by-pixel:

$$
I_{\text{corr}}(x,y) = I_{\text{LED}}(x,y) - I_{\text{bg}}(x,y) \tag{1}
$$ 


Negative values from the subtraction are clipped to zero. The image is then rotated 90° counterclockwise to match the orientation used to define the fiber-reference traces.

For each of the 80 fibers, the trace center is parametrized as a function of the horizontal camera coordinate $x$. Integration boundaries between adjacent fibers are placed halfway between neighboring trace centers, and the signal is integrated vertically over 95% of each fiber's assigned region, leaving a small dead zone between neighboring integration bands to avoid cross-talk:

$$C_{f,j} = \sum_k w_{f,j,k}\, I_{\text{corr}}(x_j, y_k) \tag{2}$$

where $w_{f,j,k}$ is the fractional (sub-pixel) contribution of vertical pixel $k$ to fiber $f$'s integration region at column $j$.

The integrated signal is converted to a count rate using the calibration-image exposure time $t_{\text{exp}}$:

$$\dot C_{f,j} = \frac{C_{f,j}}{t_{\text{exp}}} \tag{3}$$

The horizontal camera coordinate $x_j$ is converted to wavelength $\lambda_j$ using the mercury-lamp wavelength calibration, which also provides the spectral bin width $\Delta\lambda_j$ for each camera column.

## 3. Power-meter correction

The power meter is fixed at a reference setting of 750 nm, while the infrared source emits broadband. The displayed reading must be corrected for the mismatch between the detector's responsivity at 750 nm and its effective responsivity to the true broadband spectrum.

**Normalized source spectrum**, measured independently with the fiber spectrometer:

$$s(\lambda) = \frac{S(\lambda)}{\int S(\lambda')\, d\lambda'} \tag{4}$$

**Spectrum-weighted effective responsivity**, using the detector's wavelength-dependent responsivity $R(\lambda)$ [A/W]:

$$R_{\text{eff}} = \int s(\lambda)\, R(\lambda)\, d\lambda \tag{5}$$

**Corrected optical power** on the power-meter aperture:

$$P_{\text{PM},f}^{\text{true}} = P_{\text{PM},f}^{\text{display}} \cdot \frac{R(750\,\text{nm})}{R_{\text{eff}}} \tag{6}$$

**Geometric projection onto the fiber core.** The power-meter aperture (9.5 mm diameter) and the fiber core (100 µm diameter) subtend different areas:

$$\frac{A_{\text{fib}}}{A_{\text{PM}}} = \frac{\pi (d_{\text{fib}}/2)^2}{\pi (d_{\text{PM}}/2)^2} \tag{7}$$

A distance correction accounts for the fiber entrance sitting at $r_{\text{fib}} = 44\,\text{cm}$ from the source, versus the power meter measured at $r_{\text{PM}} = 39.5\,\text{cm}$. Assuming inverse-square irradiance scaling:

$$P_{\text{fib},f} = P_{\text{PM},f}^{\text{true}} \left(\frac{r_{\text{PM}}}{r_{\text{fib}}}\right)^2 \frac{A_{\text{fib}}}{A_{\text{PM}}} \tag{8}$$

$P_{\text{fib},f}$ is the estimated total broadband power crossing fiber $f$'s core. No numerical-aperture correction is applied in the current configuration — the source is treated as effectively on-axis and centered, so the fiber's acceptance cone is not a limiting factor.

## 4. Spectral power after the long-pass filter

The power meter integrates the full broadband source power, but the camera records the spectrum after the fiber and the optical filters. Only the long-pass filter is modeled explicitly; the ND filter's transmission is not separately included (see §1 and §6).

The normalized source spectrum is interpolated onto the camera wavelength axis and multiplied by the modeled long-pass transmission $T_{\text{LP}}(\lambda)$:

$$s_{\text{LP,unnorm}}(\lambda_j) = s(\lambda_j)\, T_{\text{LP}}(\lambda_j) \tag{9}$$

The current implementation uses an ideal step-function cutoff for $T_{\text{LP}}$, and additionally forces every calibrated quantity to zero below 660 nm regardless of what the transmission model would otherwise allow.

**Transmitted power fraction** — the fraction of total broadband source power that falls within the camera's spectral range:

$$F_{\text{LP}} = \sum_j s(\lambda_j)\, T_{\text{LP}}(\lambda_j)\, \Delta\lambda_j \tag{10}$$

**Total transmitted power** for fiber $f$:

$$P_{\text{LP},f} = P_{\text{fib},f} \cdot F_{\text{LP}} \tag{11}$$

**Renormalized transmitted spectral shape:**

$$s_{\text{LP}}(\lambda_j) = \frac{s(\lambda_j)\, T_{\text{LP}}(\lambda_j)}{F_{\text{LP}}} \tag{12}$$

**Known spectral power density** incident on fiber $f$:

$$P_{\lambda,f,j} = P_{\text{LP},f}\, s_{\text{LP}}(\lambda_j) \tag{13}$$

which is algebraically equivalent to

$$P_{\lambda,f,j} = P_{\text{fib},f}\, s(\lambda_j)\, T_{\text{LP}}(\lambda_j) \tag{14}$$

Equation (14) is the form actually evaluated — it avoids the intermediate normalize/renormalize round-trip of (9)–(13) while giving the same result.

## 5. Absolute calibration coefficient

The pipeline does not construct a transmission curve (measured spectrum ÷ true spectrum). It instead computes the **inverse spectral response**: known incident spectral power divided by measured count rate.

$$K_{f,j}^{\text{raw}} = \frac{P_{\lambda,f,j}}{\dot C_{f,j}} \tag{15}$$

with units

$$\left[K_{f,j}^{\text{raw}}\right] = \frac{\text{W}\,\text{nm}^{-1}}{\text{counts}\,\text{s}^{-1}} = \text{J}\,\text{nm}^{-1}\,\text{count}^{-1} \tag{16}$$

**Post-processing applied to the raw coefficients:**

- **Smoothing** — 10 nm moving average, applied in $\log_{10}(K)$ space (multiplicative noise, coefficient must stay positive).
- **Bad-fiber replacement** — manually flagged abnormal fiber curves are replaced by the wavelength-by-wavelength mean of the remaining fibers.
- **Hard zero cutoff** — $K_{f,j} \equiv 0$ for $\lambda_j < 660\,\text{nm}$, reapplied after smoothing/replacement so no interpolation step can reintroduce a nonzero coefficient below the physical long-pass cutoff.

## 6. Application to experimental shots

For a shot with background-subtracted, vertically integrated counts $C_{f,j}^{\text{shot}}$, the calibrated spectral energy density is

$$E_{\lambda,f,j} = C_{f,j}^{\text{shot}} \, K_{f,j} \tag{17}$$

in J·nm⁻¹, and the energy in wavelength bin $j$ is

$$E_{f,j}^{\text{bin}} = E_{\lambda,f,j}\, \Delta\lambda_j \tag{18}$$

in joules.

## 7. What is and isn't modeled

| Element | Treatment |
|---|---|
| Background | Subtracted explicitly (Eq. 1) |
| Fiber trace integration | Explicit, fractional-pixel weighted (Eq. 2) |
| Power-meter spectral responsivity mismatch | Explicit correction (Eqs. 4–6) |
| Aperture-area / distance geometry | Explicit (Eqs. 7–8) |
| Numerical aperture | Not applied (on-axis centered source assumption) |
| Long-pass filter | Explicit, ideal step function + hard 660 nm floor (Eqs. 9–12) |
| ND filter | **Not modeled explicitly** — absorbed into $\dot C_{f,j}$ |
| Fiber transmission, spectrometer efficiency, camera QE | **Not modeled explicitly** — absorbed into $\dot C_{f,j}$ |

Because $K_{f,j}$ is defined as a ratio of independently known incident power to measured counts, every unmodeled optical element between the fiber core and the digitized count is automatically and correctly included in the calibration, as long as it does not change between the calibration measurement and the experimental shot.
