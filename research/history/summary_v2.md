
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
| tfidf-yardstick | all | 0.951±0.003 | 0.818 | 0.992 | 97.8% | 3.92% | 0.0% | 0.00% |
| vybir-adapter | 8 | 0.842±0.068 | 0.802 | 0.899 | 68.4% | 5.50% | 47.4% | 5.65% |
| vybir-adapter | 16 | 0.888±0.023 | 0.744 | 0.923 | 83.9% | 5.19% | 31.3% | 1.70% |
| vybir-adapter | 32 | 0.893±0.026 | 0.738 | 0.929 | 84.8% | 4.98% | 45.1% | 1.81% |
| vybir-adapter | 64 | 0.910±0.023 | 0.759 | 0.934 | 89.9% | 4.72% | 34.1% | 0.63% |
| vybir-adapter | 128 | 0.925±0.014 | 0.790 | 0.940 | 93.5% | 4.92% | 37.8% | 0.41% |
| vybir-adapter | 256 | 0.934±0.010 | 0.814 | 0.947 | 95.3% | 4.76% | 39.4% | 0.46% |
| vybir-adapter | all | 0.947±0.004 | 0.849 | 0.963 | 97.5% | 4.27% | 36.9% | 0.16% |

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
| laya-zero-shot | 0 | 0.988±0.003 | 0.987 | 1.000 | 94.8% | 0.21% | 10.6% | 0.00% |
| tfidf-yardstick | 8 | 0.598±0.128 | 0.596 | 0.947 | 51.5% | 4.78% | 0.0% | 0.00% |
| tfidf-yardstick | 16 | 0.654±0.143 | 0.653 | 0.968 | 57.5% | 5.30% | 0.0% | 0.00% |
| tfidf-yardstick | 32 | 0.657±0.175 | 0.653 | 0.976 | 63.3% | 4.64% | 0.0% | 0.00% |
| tfidf-yardstick | 64 | 0.841±0.124 | 0.839 | 0.980 | 77.3% | 4.25% | 0.0% | 0.00% |
| tfidf-yardstick | 128 | 0.909±0.064 | 0.908 | 0.985 | 88.7% | 4.38% | 0.0% | 0.00% |
| tfidf-yardstick | 256 | 0.948±0.009 | 0.949 | 0.988 | 96.5% | 3.71% | 0.0% | 0.00% |
| tfidf-yardstick | all | 0.957±0.006 | 0.957 | 0.990 | 97.5% | 3.29% | 0.0% | 0.00% |
| vybir-adapter | 8 | 0.985±0.007 | 0.985 | 0.997 | 95.0% | 0.48% | 89.8% | 0.33% |
| vybir-adapter | 16 | 0.988±0.005 | 0.988 | 0.997 | 95.1% | 0.39% | 85.3% | 0.34% |
| vybir-adapter | 32 | 0.992±0.005 | 0.992 | 0.999 | 95.5% | 0.24% | 80.7% | 0.23% |
| vybir-adapter | 64 | 0.992±0.005 | 0.992 | 0.999 | 95.1% | 0.23% | 77.3% | 0.27% |
| vybir-adapter | 128 | 0.994±0.003 | 0.994 | 0.999 | 95.1% | 0.15% | 86.3% | 0.12% |
| vybir-adapter | 256 | 0.994±0.003 | 0.994 | 1.000 | 94.8% | 0.07% | 87.6% | 0.09% |
| vybir-adapter | all | 0.994±0.003 | 0.994 | 1.000 | 95.2% | 0.08% | 89.2% | 0.06% |

## Guarantee check: P(automatic and wrong) must stay <= alpha

| task | method | n | alpha | coverage | auto | joint err (mean ± s.e.) | seeds over alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| sms_scam | laya-zero-shot | 0 | 0.01 | 0.990 | 21.0% | 0.95% ± 0.15% | 8/20 |
| sms_scam | laya-zero-shot | 0 | 0.05 | 0.947 | 51.8% | 5.31% ± 0.31% | 14/20 |
| sms_scam | laya-zero-shot | 0 | 0.1 | 0.895 | 69.7% | 10.46% ± 0.45% | 11/20 |
| sms_scam | laya-zero-shot | 0 | 0.05 (PAC, delta=0.1) | 0.972 | 39.0% | 2.79% ± 0.23% | 1/20 |
| sms_scam | vybir-adapter | 64 | 0.01 | 0.989 | 59.0% | 1.11% ± 0.15% | 9/20 |
| sms_scam | vybir-adapter | 64 | 0.05 | 0.952 | 89.9% | 4.72% ± 0.26% | 8/20 |
| sms_scam | vybir-adapter | 64 | 0.1 | 0.903 | 95.7% | 7.01% ± 0.50% | 1/20 |
| sms_scam | vybir-adapter | 64 | 0.05 (PAC, delta=0.1) | 0.969 | 80.6% | 3.06% ± 0.28% | 1/20 |
| sms_scam | vybir-adapter | 3144 | 0.01 | 0.988 | 73.6% | 1.20% ± 0.21% | 10/20 |
| sms_scam | vybir-adapter | 3144 | 0.05 | 0.949 | 97.5% | 4.27% ± 0.18% | 5/20 |
| sms_scam | vybir-adapter | 3144 | 0.1 | 0.901 | 92.7% | 2.66% ± 0.16% | 0/20 |
| sms_scam | vybir-adapter | 3144 | 0.05 (PAC, delta=0.1) | 0.966 | 94.0% | 3.21% ± 0.24% | 1/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.01 | 0.989 | 86.8% | 1.07% ± 0.14% | 9/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.05 | 0.949 | 97.8% | 3.92% ± 0.18% | 1/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.1 | 0.903 | 92.4% | 2.07% ± 0.19% | 0/20 |
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
| jailbreak | laya-zero-shot | 0 | 0.01 | 0.988 | 99.0% | 0.82% ± 0.10% | 6/20 |
| jailbreak | laya-zero-shot | 0 | 0.05 | 0.946 | 94.8% | 0.21% ± 0.03% | 0/20 |
| jailbreak | laya-zero-shot | 0 | 0.1 | 0.905 | 90.6% | 0.11% ± 0.03% | 0/20 |
| jailbreak | laya-zero-shot | 0 | 0.05 (PAC, delta=0.1) | 0.966 | 96.9% | 0.32% ± 0.05% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.01 | 0.989 | 98.9% | 0.56% ± 0.09% | 3/20 |
| jailbreak | vybir-adapter | 64 | 0.05 | 0.949 | 95.1% | 0.23% ± 0.05% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.1 | 0.906 | 90.7% | 0.11% ± 0.04% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.05 (PAC, delta=0.1) | 0.969 | 97.3% | 0.39% ± 0.08% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.01 | 0.991 | 99.1% | 0.40% ± 0.05% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.05 | 0.951 | 95.2% | 0.08% ± 0.02% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.1 | 0.909 | 90.9% | 0.03% ± 0.02% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.05 (PAC, delta=0.1) | 0.971 | 97.3% | 0.23% ± 0.03% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.01 | 0.988 | 84.2% | 1.22% ± 0.13% | 13/20 |
| jailbreak | tfidf-yardstick | 584 | 0.05 | 0.949 | 97.5% | 3.29% ± 0.17% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.1 | 0.898 | 91.9% | 2.08% ± 0.13% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.05 (PAC, delta=0.1) | 0.968 | 96.3% | 3.14% ± 0.25% | 2/20 |