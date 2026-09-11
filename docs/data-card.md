# Data dictionary - features.csv

Generated from the data itself by `make pipeline`; do not edit by hand.

| Column | Type | Null rate | Distinct | Min | Median | Max |
|---|---|---|---|---|---|---|
| `RAM_SIZE` | float64 | 0.5% | 22 | 1.0 | 16.0 | 128.0 |
| `SSD_SIZE` | float64 | 0.0% | 108 | 0.0 | 256.0 | 12800.0 |
| `HDD_SIZE` | float64 | 0.0% | 75 | 0.0 | 0.0 | 10000.0 |
| `RAM_TYPE` | float64 | 4.4% | 10 | 0.5 | 3.0 | 4.2 |
| `cores` | float64 | 4.8% | 9 | 1.0 | 4.0 | 16.0 |
| `cpu_mark` | float64 | 4.8% | 629 | 123.0 | 10010.0 | 57591.0 |
| `tdp` | float64 | 4.8% | 46 | 2.5 | 15.0 | 148.0 |
| `gpu_g3d_mark` | float64 | 4.0% | 122 | 4.0 | 1042.0 | 60000.0 |
| `gpu_g2d_mark` | float64 | 4.0% | 115 | 23.0 | 300.0 | 1800.0 |
| `gpu_tdp` | float64 | 4.0% | 55 | 6.0 | 20.0 | 575.0 |
| `SCREEN_SIZE_SNAPPED` | float64 | 0.0% | 30 | 10.0 | 14.0 | 20.0 |
| `SCREEN_RESOLUTION_ENC` | float64 | 2.2% | 9 | 1.0 | 3.0 | 9.0 |
| `cpu_generation_normalized` | float64 | 8.6% | 23 | 0.0 | 0.667 | 1.0 |
| `spec_Etat` | float64 | 41.4% | 3 | 1.0 | 2.0 | 3.0 |
| `gpu_to_cpu_ratio` | float64 | 4.9% | 1,412 | 0.001 | 0.196 | 11.861 |
| `storage_per_ram` | float64 | 0.5% | 147 | 0.0 | 32.0 | 3200.0 |
| `total_storage` | float64 | 0.0% | 166 | 0.0 | 512.0 | 12800.0 |
| `pixels` | float64 | 2.2% | 9 | 1049088.0 | 2073600.0 | 14745600.0 |
| `ppi` | float64 | 2.2% | 95 | 90.583 | 157.351 | 376.565 |
| `total_tdp` | float64 | 0.0% | 166 | 0.0 | 40.0 | 630.0 |
| `perf_per_expected_dinar` | float64 | 4.8% | 5,629 | 0.002 | 0.094 | 0.34 |
| `estimated_component_cost` | float64 | 0.0% | 2,623 | 7600.0 | 94200.0 | 2252916000.0 |
| `listing_year` | int64 | 0.0% | 8 | 2018.0 | 2025.0 | 2025.0 |
| `listing_month` | int64 | 0.0% | 12 | 1.0 | 7.0 | 12.0 |
| `listing_month_index` | int64 | 0.0% | 80 | 11.0 | 88.0 | 92.0 |
| `month_sin` | float64 | 0.0% | 11 | -1.0 | -0.5 | 1.0 |
| `month_cos` | float64 | 0.0% | 11 | -1.0 | -0.5 | 1.0 |
| `etat_is_missing` | int64 | 0.0% | 2 | 0.0 | 0.0 | 1.0 |
| `has_hdd` | int64 | 0.0% | 2 | 0.0 | 0.0 | 1.0 |
| `is_dual_drive` | int64 | 0.0% | 2 | 0.0 | 0.0 | 1.0 |
| `has_dedicated_gpu` | int64 | 0.0% | 2 | 0.0 | 0.0 | 1.0 |
| `model_family` | int64 | 0.0% | 4 | 0.0 | 1.0 | 3.0 |
| `brand` | object | 0.0% | 39 |  | mode=THINKPAD |  |
| `city_grouped` | object | 0.0% | 59 |  | mode=EZZOUAR |  |
| `cpu_manufacturer` | object | 0.0% | 5 |  | mode=Intel |  |
| `cpu_family` | object | 0.0% | 35 |  | mode=Intel_i5 |  |
| `price_corrected` | float64 | 0.0% | 856 | 10000.0 | 95000.0 | 1000000.0 |
| `spec_signature` | object | 0.0% | 6,944 |  | mode=8.0|256.0|0.0|5800|1042|14.0|3.0 |  |
| `price_unit_ambiguous` | bool | 0.0% | 2 | nan | 0.0 | nan |
