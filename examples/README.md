# Notes 
1. Examples only run 1000 Langevin steps to test performance.
2. The current neural networks are trained with the predictor-corrector method (except Gyroid). They need to be retrained with the Leimkuhler-Matthews method.
3. For BottleBrush and Star9ArmsGyroid, the domain sizes are not accurately equilibrated, and there will be an error within 5%. 
4. For BottleBrush and Star9ArmsGyroid, the grids number are not multiple of *2^n*. There could be performance drop of FFT due to undesirable grid size.
