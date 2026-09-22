
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
| vybir-adapter | 8 | 0.838±0.065 | 0.801 | 0.896 | 67.3% | 5.42% | 50.7% | 5.67% |
| vybir-adapter | 16 | 0.889±0.024 | 0.767 | 0.914 | 83.6% | 5.22% | 40.5% | 1.77% |
| vybir-adapter | 32 | 0.895±0.025 | 0.753 | 0.928 | 84.3% | 4.94% | 47.2% | 1.83% |
| vybir-adapter | 64 | 0.911±0.022 | 0.764 | 0.934 | 89.9% | 4.75% | 35.4% | 0.63% |
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
| vybir-adapter | 8 | 0.757±0.045 | 0.729 | 0.830 | 41.7% | 4.83% | 21.3% | 3.21% |
| vybir-adapter | 16 | 0.768±0.028 | 0.738 | 0.842 | 44.8% | 5.26% | 8.1% | 0.74% |
| vybir-adapter | 32 | 0.777±0.037 | 0.747 | 0.847 | 48.1% | 5.19% | 7.3% | 0.38% |
| vybir-adapter | 64 | 0.787±0.031 | 0.759 | 0.859 | 50.2% | 5.04% | 4.4% | 0.09% |
| vybir-adapter | 128 | 0.796±0.020 | 0.770 | 0.864 | 49.3% | 4.51% | 4.4% | 0.06% |
| vybir-adapter | 197 | 0.797±0.022 | 0.773 | 0.864 | 50.2% | 4.96% | 5.3% | 0.09% |

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
| vybir-adapter | 8 | 0.985±0.007 | 0.985 | 0.997 | 95.1% | 0.51% | 89.4% | 0.33% |
| vybir-adapter | 16 | 0.988±0.005 | 0.988 | 0.997 | 95.1% | 0.42% | 86.2% | 0.34% |
| vybir-adapter | 32 | 0.992±0.005 | 0.992 | 0.999 | 95.4% | 0.25% | 81.6% | 0.23% |
| vybir-adapter | 64 | 0.992±0.005 | 0.992 | 0.999 | 95.0% | 0.23% | 79.1% | 0.27% |
| vybir-adapter | 128 | 0.994±0.003 | 0.994 | 0.999 | 95.1% | 0.15% | 86.3% | 0.12% |
| vybir-adapter | 256 | 0.994±0.003 | 0.994 | 1.000 | 94.8% | 0.07% | 87.6% | 0.09% |
| vybir-adapter | all | 0.994±0.003 | 0.994 | 1.000 | 95.2% | 0.08% | 89.2% | 0.06% |

## Guarantee check: P(automatic and wrong) must stay <= alpha

| task | method | n | alpha | coverage | auto | joint err (mean ± s.e.) | seeds over alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| sms_scam | laya-zero-shot | 0 | 0.01 | 0.990 | 21.0% | 0.95% ± 0.15% | 8/20 |
| sms_scam | laya-zero-shot | 0 | 0.05 | 0.947 | 51.8% | 5.31% ± 0.31% | 14/20 |
| sms_scam | laya-zero-shot | 0 | 0.1 | 0.895 | 69.7% | 10.46% ± 0.45% | 11/20 |
| sms_scam | vybir-adapter | 64 | 0.01 | 0.989 | 59.0% | 1.13% ± 0.16% | 9/20 |
| sms_scam | vybir-adapter | 64 | 0.05 | 0.951 | 89.9% | 4.75% ± 0.29% | 8/20 |
| sms_scam | vybir-adapter | 64 | 0.1 | 0.902 | 95.8% | 7.06% ± 0.53% | 1/20 |
| sms_scam | vybir-adapter | 3144 | 0.01 | 0.988 | 73.6% | 1.20% ± 0.21% | 10/20 |
| sms_scam | vybir-adapter | 3144 | 0.05 | 0.949 | 97.5% | 4.27% ± 0.18% | 5/20 |
| sms_scam | vybir-adapter | 3144 | 0.1 | 0.901 | 92.7% | 2.66% ± 0.16% | 0/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.01 | 0.989 | 86.8% | 1.07% ± 0.14% | 9/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.05 | 0.949 | 97.8% | 3.92% ± 0.18% | 1/20 |
| sms_scam | tfidf-yardstick | 3144 | 0.1 | 0.903 | 92.4% | 2.07% ± 0.19% | 0/20 |
| prompt_injection | laya-zero-shot | 0 | 0.01 | 0.992 | 13.5% | 0.83% ± 0.15% | 8/20 |
| prompt_injection | laya-zero-shot | 0 | 0.05 | 0.945 | 48.8% | 5.53% ± 0.46% | 12/20 |
| prompt_injection | laya-zero-shot | 0 | 0.1 | 0.895 | 65.9% | 10.51% ± 0.61% | 14/20 |
| prompt_injection | vybir-adapter | 64 | 0.01 | 0.991 | 15.2% | 0.94% ± 0.19% | 8/20 |
| prompt_injection | vybir-adapter | 64 | 0.05 | 0.950 | 50.2% | 5.04% ± 0.44% | 10/20 |
| prompt_injection | vybir-adapter | 64 | 0.1 | 0.896 | 73.3% | 10.43% ± 0.65% | 13/20 |
| prompt_injection | vybir-adapter | 197 | 0.01 | 0.989 | 17.9% | 1.08% ± 0.17% | 11/20 |
| prompt_injection | vybir-adapter | 197 | 0.05 | 0.950 | 50.2% | 4.96% ± 0.57% | 7/20 |
| prompt_injection | vybir-adapter | 197 | 0.1 | 0.901 | 71.9% | 9.91% ± 0.72% | 11/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.01 | 0.988 | 24.6% | 1.19% ± 0.18% | 11/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.05 | 0.955 | 49.8% | 4.45% ± 0.42% | 7/20 |
| prompt_injection | tfidf-yardstick | 197 | 0.1 | 0.903 | 68.8% | 9.70% ± 0.67% | 11/20 |
| jailbreak | laya-zero-shot | 0 | 0.01 | 0.988 | 99.0% | 0.82% ± 0.10% | 6/20 |
| jailbreak | laya-zero-shot | 0 | 0.05 | 0.946 | 94.8% | 0.21% ± 0.03% | 0/20 |
| jailbreak | laya-zero-shot | 0 | 0.1 | 0.905 | 90.6% | 0.11% ± 0.03% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.01 | 0.989 | 98.9% | 0.56% ± 0.09% | 3/20 |
| jailbreak | vybir-adapter | 64 | 0.05 | 0.948 | 95.0% | 0.23% ± 0.05% | 0/20 |
| jailbreak | vybir-adapter | 64 | 0.1 | 0.906 | 90.7% | 0.11% ± 0.04% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.01 | 0.991 | 99.1% | 0.40% ± 0.05% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.05 | 0.951 | 95.2% | 0.08% ± 0.02% | 0/20 |
| jailbreak | vybir-adapter | 584 | 0.1 | 0.909 | 90.9% | 0.03% ± 0.02% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.01 | 0.988 | 84.2% | 1.22% ± 0.13% | 13/20 |
| jailbreak | tfidf-yardstick | 584 | 0.05 | 0.949 | 97.5% | 3.29% ± 0.17% | 0/20 |
| jailbreak | tfidf-yardstick | 584 | 0.1 | 0.898 | 91.9% | 2.08% ± 0.13% | 0/20 |