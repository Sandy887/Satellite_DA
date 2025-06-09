# Error Model

A preliminary code release of a cloud-aware error model.

## Overview

Observation-minus-background (O–B) departures often deviate from a Gaussian distribution under cloudy conditions—violating the core Gaussianity assumption in many data assimilation systems. Major forecasting centers (e.g. ECMWF) have studied this challenge using all-sky infrared and microwave channels, developing methods to “Gaussianize” the errors.

This toolkit applies a similar idea to the SEVIRI 0.6 µm channel (visible reflectance). By quantifying cloud amount—using both synthetic and observed cloud amount/effect and incorporating it into the error model, I more accurately capture the true shape of the O–B distribution. Although reflectance values are naturally bounded (and thus exhibit their own non-Gaussian behavior), I use the 0.6 µm channel here as a clear, illustrative example of the cloud-dependent approach.  

## Files

See **CONTENTS.md** for a complete list and descriptions.

## Getting Started

1. **Clone or download** this repository.  
2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib scipy
3. **Explore the example notebook:**
   ```bash
   jupyter notebook run_error_model.ipynb

## Acknowledgments
- Refactored from my original code with the help of GitHub Copilot.

- Contributions and feedback welcome!

## License

© 2025 Sandy Chkeir. Released under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your work, please cite:

> Chkeir, S. (2025). Error Model Utils: A cloud-dependent error model (Version 1.0) [Source code]. GitHub. https://github.com/Sandy887/Satellite_DA/Non-Gaussian-errors

### BibTex:
```bibtex
@software{error_model_utils_2025,
  author    = {Chkeir, S.},
  title     = {Non-Gaussian-errors: A cloud-dependent error model},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Sandy887/Satellite_DA/Non-Gaussian-errors},
  note      = {Source code}
}