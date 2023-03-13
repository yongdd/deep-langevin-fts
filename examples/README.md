# Note 
1. Examples only run 1000 Langevin steps to test performance.
2. The current neural networks are trained with the predictor-corrector method. They need to be retrained with the Leimkuhler-Matthews method.
3. For BottleBrush and Star9ArmsGyroid, the domain sizes are not accurately equilibrated, and there will be an error within 5%. 
4. For BottleBrush and Star9ArmsGyroid, the grids number are not *2^n*. There will be performance drop of FFT due to undesirable grid size.
