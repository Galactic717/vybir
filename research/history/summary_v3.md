
### sms_scam  (5574 examples, 747 positive)

| method | n labels | accuracy | balanced acc | AUROC | auto @ a=0.05 | joint err @ a=0.05 | naive p>=0.99 auto | naive joint err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| laya-zero-shot | 0 | 0.766±0.008 | 0.847 | 0.932 | 51.8% | 5.31% | 0.0% | 0.00% |
| tfidf-yardstick | 8 | 0.866±0.001 | 0.515 | 0.789 | 70.4% | 4.63% | 0.0% | 0.00% |
| tfidf-yardstick | 16 | 0.867±0.004 | 0.505 | 0.846 | 78.8% | 4.78% | 0.0% | 0.00% |
| tfidf-yardstick | 32 | 0.866±0.000 | 0.500 | 0.908 | 86.3% | 5.04% | 0.0% | 0.00% |
| tfidf-yardstick | 64 | 0.866±0.000 | 0.500 | 0.950 | 89.1% | 4.73% | 0.0% | 0.00% |
| tfidf-yardstick | 128 | 0.866±0.000 | 0.500 | 0.971 | 90.9% | 4.99% | 0.0% | 0.00% |
| tfidf-yardstick | 256 | 0.866±0.000 | 0.500 | 0.982 | 91.4% | 5.12% | 0.0% | 0.00% |
| tfidf-yardstick | all | 0.951±0.003 | 0.818 | 0.992 | 98.9% | 4.43% | 0.0% | 0.00% |
| vybir-adapter | 8 | 0.842±0.068 | 0.802 | 0.899 | 68.4% | 5.50% | 47.4% | 5.65% |
| vybir-adapter | 16 | 0.888±0.023 | 0.744 | 0.923 | 83.9% | 5.19% | 31.3% | 1.70% |
| vybir-adapter | 32 | 0.893±0.026 | 0.738 | 0.929 | 84.8% | 4.98% | 45.1% | 1.81% |
| vybir-adapter | 64 | 0.910±0.023 | 0.759 | 0.934 | 90.0% | 4.78% | 34.1% | 0.63% |
| vybir-adapter | 128 | 0.925±0.014 | 0.790 | 0.940 | 93.9% | 5.11% | 37.8% | 0.41% |
| vybir-adapter | 256 | 0.934±0.010 | 0.814 | 0.947 | 95.6% | 4.91% | 39.4% | 0.46% |
| vybir-adapter | all | 0.947±0.004 | 0.849 | 0.963 | 98.3% | 4.61% | 36.9% | 0.16% |

### prompt_injection  (662 examples, 263 positive)

| method | n labels | accuracy | balanced acc | AUROC | auto @ a=0.05 | joint err @ a=0.05 | naive p>=0.99 auto | naive joint err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| laya-zero-shot | 0 | 0.760±0.020 | 0.707 | 0.851 | 48.8% | 5.53% | 0.8% | 0.00% |
| tfidf-yardstick | 8 | 0.595±0.043 | 0.535 | 0.637 | 16.7% | 4.23% | 0.0% | 0.00% |
| tfidf-yardstick | 16 | 0.590±0.061 | 0.531 | 0.682 | 18.8% | 4.26% | 0.0% | 0.00% |
| tfidf-yardstick | 32 | 0.612±0.056 | 0.549 | 0.746 | 26.9% | 4.77% | 0.0% | 0.00% |
| tfidf-yardstick | 64 | 0.671±0.071 | 0.588 | 0.819 | 37.7% | 4.89% | 0.0% | 0.00% |
| tfidf-yardstick | 128 | 0.718±0.066 | 0.648 | 0.880 | 48.3% | 5.25% | 0.0% | 0.00% |
| tfidf-yardstick | 197 | 0.760±0.043 | 0.700 | 0.906 | 49.8% | 4.45% | 0.0% | 0.00% |
| vybir-adapter | 8 | 0.773±0.039 | 0.746 | 0.845 | 48.1% | 5.25% | 17.7% | 2.83% |
| vybir-adapter | 16 | 0.779±0.022 | 0.750 | 0.848 | 48.5% | 5.43% | 7.0% | 0.64% |
| vybir-adapter | 32 | 0.778±0.036 | 0.748 | 0.844 | 49.2% | 5.34% | 6.6% | 0.32% |
| vybir-adapter | 64 | 0.786±0.029 | 0.757 | 0.856 | 50.8% | 5.13% | 4.0% | 0.09% |
| vybir-adapter | 128 | 0.794±0.021 | 0.768 | 0.862 | 48.4% | 4.38% | 4.2% | 0.06% |
| vybir-adapter | 197 | 0.797±0.022 | 0.772 | 0.863 | 50.0% | 4.94% | 5.3% | 0.09% |

### jailbreak  (1306 examples, 666 positive)

| method | n labels | accuracy | balanced acc | AUROC | auto @ a=0.05 | joint err @ a=0.05 | naive p>=0.99 auto | naive joint err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| laya-zero-shot | 0 | 0.988±0.003 | 0.987 | 1.000 | 100.0% | 1.25% | 10.6% | 0.00% |
| tfidf-yardstick | 8 | 0.598±0.128 | 0.596 | 0.947 | 51.5% | 4.78% | 0.0% | 0.00% |
| tfidf-yardstick | 16 | 0.654±0.143 | 0.653 | 0.968 | 57.5% | 5.30% | 0.0% | 0.00% |
| tfidf-yardstick | 32 | 0.657±0.175 | 0.653 | 0.976 | 63.3% | 4.64% | 0.0% | 0.00% |
| tfidf-yardstick | 64 | 0.841±0.124 | 0.839 | 0.980 | 77.8% | 4.46% | 0.0% | 0.00% |
| tfidf-yardstick | 128 | 0.909±0.064 | 0.908 | 0.985 | 89.4% | 4.66% | 0.0% | 0.00% |
| tfidf-yardstick | 256 | 0.948±0.009 | 0.949 | 0.988 | 97.7% | 4.21% | 0.0% | 0.00% |
| tfidf-yardstick | all | 0.957±0.006 | 0.957 | 0.990 | 99.3% | 4.02% | 0.0% | 0.00% |
| vybir-adapter | 8 | 0.985±0.007 | 0.985 | 0.997 | 100.0% | 1.46% | 89.8% | 0.33% |
| vybir-adapter | 16 | 0.988±0.005 | 0.988 | 0.997 | 100.0% | 1.17% | 85.3% | 0.34% |
| vybir-adapter | 32 | 0.992±0.005 | 0.992 | 0.999 | 100.0% | 0.82% | 80.7% | 0.23% |
| vybir-adapter | 64 | 0.992±0.005 | 0.992 | 0.999 | 100.0% | 0.84% | 77.3% | 0.27% |
| vybir-adapter | 128 | 0.994±0.003 | 0.994 | 0.999 | 100.0% | 0.64% | 86.3% | 0.12% |
| vybir-adapter | 256 | 0.994±0.003 | 0.994 | 1.000 | 100.0% | 0.63% | 87.6% | 0.09% |
| vybir-adapter | all | 0.994±0.003 | 0.994 | 1.000 | 100.0% | 0.56% | 89.2% | 0.06% |

## Guarantee check: P(automatic and wrong) must stay <= alpha

| task | method | n | alpha | coverage | auto | joint err (mean ± s.e.) | seeds over alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| sms_scam | laya-zero-shot | 0 | 0.01 | 0.990 | 21.0% | 0.95% ± 0.15% | 8/20 |
| sms_scam | laya-zero-shot | 0 | 0.05 | 0.947 | 51.8% | 5.31% ± 0.31% | 14/20 |
| sms_scam | laya-zero-shot | 0 | 0.1 | 0.895 | 69.7% | 10.46% ± 0.45% | 11/20 |
| sms_scam | laya-zero-shot | 0 | 0.05 (PAC, delta=0.1) | 0.972 | 39.0% | 2.79% ± 0.23% | 1/20 |
| sms_scam | vybir-adapter | 64 | 0.01 | 0.989 | 59.0% | 1.11% ± 0.15% | 9/20 |
| sms_scam | vybir-adapter | 64 | 0.05 | 0.952 | 90.0% | 4.78% ± 0.28% | 8/20 |
| sms_scam | vybir-adapter | 64 | 0.1 | 0.903 | 98.4% | 8.05% ± 0.35% | 1/20 |
| sms_scam | vybir-adapter | 64 | 0.05 (PAC, delta=0.1) | 0.969 | 80.6% | 3.06% ± 0.28% | 1/20 |
| sms_scam | vybir-adapter | 3144 | 0.01 | 0.988 | 73.6% | 1.20% ± 0.21% | 10/20 |
| sms_scam | vybir-adapter | 3144 | 0.05 | 0.949 | 98.3% | 4.61% ± 0.18% | 9/20 |
| sms_scam | vybir-adapter | 3144 | 0.1 | 0.901 | 100.0% | 5.35% ± 0.08% | 0/20 |
| sms_scam | vybir-adapter | 3144 | 0.05 (PAC, delta=0.1) | 0.966 | 94.1% | 3.29% ± 0.26% | 2/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.01 | 0.989 | 86.8% | 1.07% ± 0.14% | 9/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.05 | 0.949 | 98.9% | 4.43% ± 0.16% | 5/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.1 | 0.903 | 100.0% | 4.89% ± 0.07% | 0/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.05 (PAC, delta=0.1) | 0.973 | 94.5% | 2.72% ± 0.25% | 0/20 |
| prompt_injection | laya-zero-shot | 0 | 0.01 | 0.992 | 13.5% | 0.83% ± 0.15% | 8/20 |
| prompt_injection | laya-zero-shot | 0 | 0.05 | 0.945 | 48.8% | 5.53% ± 0.46% | 12/20 |
| prompt_injection | laya-zero-shot | 0 | 0.1 | 0.895 | 65.9% | 10.51% ± 0.61% | 14/20 |
| prompt_injection | laya-zero-shot | 0 | 0.05 (PAC, delta=0.1) | 0.964 | 37.5% | 3.60% ± 0.40% | 3/20 |
| prompt_injection | vybir-adapter | 64 | 0.01 | 0.991 | 14.2% | 0.89% ± 0.18% | 7/20 |
| prompt_injection | vybir-adapter | 64 | 0.05 | 0.949 | 50.8% | 5.13% ± 0.49% | 11/20 |
| prompt_injection | vybir-adapter | 64 | 0.1 | 0.897 | 72.6% | 10.30% ± 0.64% | 13/20 |
| prompt_injection | vybir-adapter | 64 | 0.05 (PAC, delta=0.1) | 0.971 | 32.3% | 2.92% ± 0.34% | 3/20 |
| prompt_injection | vybir-adapter | 197 | 0.01 | 0.989 | 17.3% | 1.06% ± 0.17% | 10/20 |
| prompt_injection | vybir-adapter | 197 | 0.05 | 0.951 | 50.0% | 4.94% ± 0.57% | 7/20 |
| prompt_injection | vybir-adapter | 197 | 0.1 | 0.901 | 71.9% | 9.94% ± 0.72% | 11/20 |
| prompt_injection | vybir-adapter | 197 | 0.05 (PAC, delta=0.1) | 0.973 | 35.3% | 2.74% ± 0.42% | 3/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.01 | 0.988 | 24.6% | 1.19% ± 0.18% | 11/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.05 | 0.955 | 49.8% | 4.45% ± 0.42% | 7/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.1 | 0.903 | 68.8% | 9.70% ± 0.67% | 11/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.05 (PAC, delta=0.1) | 0.973 | 41.6% | 2.72% ± 0.32% | 1/20 |
| jailbreak | laya-zero-shot | 0 | 0.01 | 0.988 | 99.4% | 1.00% ± 0.11% | 9/20 |
| jailbreak | laya-zero-shot | 0 | 0.05 | 0.946 | 100.0% | 1.25% ± 0.08% | 0/20 |
| jailbreak | laya-zero-shot | 0 | 0.1 | 0.905 | 100.0% | 1.25% ± 0.08% | 0/20 |
| jailbreak | laya-zero-shot | 0 | 0.05 (PAC, delta=0.1) | 0.966 | 100.0% | 1.25% ± 0.08% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.01 | 0.989 | 99.5% | 0.72% ± 0.11% | 3/20 |
| jailbreak | vybir-adapter | 64 | 0.05 | 0.949 | 100.0% | 0.84% ± 0.11% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.1 | 0.906 | 100.0% | 0.84% ± 0.11% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.05 (PAC, delta=0.1) | 0.969 | 100.0% | 0.84% ± 0.11% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.01 | 0.991 | 99.7% | 0.53% ± 0.07% | 1/20 |
| jailbreak | vybir-adapter | 584 | 0.05 | 0.951 | 100.0% | 0.56% ± 0.06% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.1 | 0.909 | 100.0% | 0.56% ± 0.06% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.05 (PAC, delta=0.1) | 0.971 | 100.0% | 0.56% ± 0.06% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.01 | 0.988 | 84.2% | 1.22% ± 0.13% | 13/20 |
| jailbreak | tfidf-yardstick | 584 | 0.05 | 0.949 | 99.3% | 4.02% ± 0.19% | 2/20 |
| jailbreak | tfidf-yardstick | 584 | 0.1 | 0.898 | 100.0% | 4.34% ± 0.14% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.05 (PAC, delta=0.1) | 0.968 | 96.4% | 3.21% ± 0.26% | 2/20 |