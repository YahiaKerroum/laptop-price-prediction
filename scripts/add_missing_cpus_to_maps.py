"""
Add Missing CPUs to Mapping Files
==================================
This script adds missing CPUs found in the data to cpu_ddr_map.csv and cpu_storage_map.csv
with appropriate DDR types and storage configurations based on their generation and tier.
"""

import csv
import re
from pathlib import Path

# Missing CPUs to add with their DDR types and storage configs
MISSING_CPUS_DDR = {
    # AMD CPUs
    'AMD A4-3320M APU': ('DDR3', 2011, 'Llano APU'),
    'AMD Athlon 220GE': ('DDR4', 2019, 'Zen APU'),
    'AMD E-350': ('DDR3', 2011, 'Bobcat APU'),
    'AMD Embedded G-Series GX-215JJ Radeon R2E': ('DDR3', 2014, 'Embedded APU'),
    'AMD Ryzen 3 210': ('DDR5', 2024, 'Zen 4 entry'),
    'AMD Ryzen 3 7335U': ('DDR5', 2023, 'Zen 3+ Rembrandt-R'),
    'AMD Ryzen 5 240': ('DDR5', 2024, 'Zen 4'),
    'AMD Ryzen 5 2600H': ('DDR4', 2018, 'Zen+ mobile H'),
    'AMD Ryzen 5 3500C': ('DDR4', 2020, 'Zen+ Chromebook'),
    'AMD Ryzen 5 40': ('DDR5', 2024, 'Zen 4'),
    'AMD Ryzen 5 7235HS': ('DDR5', 2023, 'Zen 3+ Rembrandt'),
    'AMD Ryzen 5 7545U': ('DDR5', 2023, 'Zen 4 Phoenix'),
    'AMD Ryzen 5 8540U': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 5 8640HS': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 5 8645HS': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 5 PRO 2500U': ('DDR4', 2018, 'Zen+ PRO mobile'),
    'AMD Ryzen 5 PRO 4500U': ('DDR4', 2020, 'Zen 2 PRO Renoir'),
    'AMD Ryzen 5 PRO 8540U': ('DDR5', 2024, 'Zen 4 PRO Hawk Point'),
    'AMD Ryzen 7 3700C': ('DDR4', 2020, 'Zen+ Chromebook'),
    'AMD Ryzen 7 7435HS': ('DDR5', 2023, 'Zen 3+ Rembrandt'),
    'AMD Ryzen 7 8840U': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 7 8845H': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 7 8845HS': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 7 PRO 2700U': ('DDR4', 2018, 'Zen+ PRO mobile'),
    'AMD Ryzen 7 PRO 3700U': ('DDR4', 2019, 'Zen+ PRO Picasso'),
    'AMD Ryzen 7 PRO 7730U': ('DDR4', 2022, 'Zen 3+ PRO Barcelo'),
    'AMD Ryzen 7 PRO 7840HS': ('DDR5', 2023, 'Zen 4 PRO Phoenix'),
    'AMD Ryzen 7 PRO 8840HS': ('DDR5', 2024, 'Zen 4 PRO Hawk Point'),
    'AMD Ryzen 7 PRO 8840U': ('DDR5', 2024, 'Zen 4 PRO Hawk Point'),
    'AMD Ryzen 9 7940HX': ('DDR5', 2023, 'Zen 4 Dragon Range'),
    'AMD Ryzen 9 8945H': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 9 8945HS': ('DDR5', 2024, 'Zen 4 Hawk Point'),
    'AMD Ryzen 9 9955HX': ('DDR5', 2024, 'Zen 5 Strix Halo'),
    'AMD Ryzen AI 9 HX 370': ('DDR5', 2024, 'Zen 5 Strix Point'),
    'AMD Sempron M120': ('DDR2', 2008, 'K8 mobile'),
    
    # Apple Silicon
    'Apple M1 8 Core 3200 MHz': ('LPDDR4X', 2020, 'Apple Silicon M1'),
    'Apple M1 Max 10 Core 3200 MHz': ('LPDDR5', 2021, 'Apple Silicon M1 Max'),
    'Apple M1 Pro 10 Core 3200 MHz': ('LPDDR5', 2021, 'Apple Silicon M1 Pro'),
    'Apple M2 8 Core 3500 MHz': ('LPDDR5', 2022, 'Apple Silicon M2'),
    'Apple M2 Pro 10 Core 3480 MHz': ('LPDDR5', 2023, 'Apple Silicon M2 Pro'),
    'Apple M3 8 Core': ('LPDDR5', 2023, 'Apple Silicon M3'),
    'Apple M3 Max 14 Core': ('LPDDR5', 2023, 'Apple Silicon M3 Max'),
    'Apple M3 Pro 11 Core': ('LPDDR5', 2023, 'Apple Silicon M3 Pro'),
    'Apple M4 10 Core': ('LPDDR5X', 2024, 'Apple Silicon M4'),
    'Apple M4 Max 14 Core': ('LPDDR5X', 2024, 'Apple Silicon M4 Max'),
    'Apple M4 Pro 12 Core': ('LPDDR5X', 2024, 'Apple Silicon M4 Pro'),
    
    # Intel Celeron
    'Celeron Dual-Core T3000': ('DDR2', 2008, 'Merom Celeron'),
    'Celeron Dual-Core T3100': ('DDR2', 2008, 'Penryn Celeron'),
    'Intel Atom C2338': ('DDR3', 2013, 'Avoton server'),
    'Intel Atom x5-Z8300': ('DDR3L', 2015, 'Cherry Trail tablet'),
    'Intel Celeron 2955U': ('DDR3', 2013, 'Haswell ULT'),
    'Intel Celeron 2957U': ('DDR3', 2013, 'Haswell ULT'),
    'Intel Celeron 2980U': ('DDR3', 2014, 'Haswell ULT'),
    'Intel Celeron 3867U': ('DDR4', 2017, 'Kaby Lake'),
    'Intel Celeron 6305': ('DDR4', 2021, 'Tiger Lake'),
    'Intel Celeron 7305': ('DDR4', 2022, 'Alder Lake'),
    'Intel Celeron B815': ('DDR3', 2011, 'Sandy Bridge mobile'),
    'Intel Celeron B840': ('DDR3', 2012, 'Sandy Bridge mobile'),
    'Intel Celeron M 900MHz': ('DDR', 2004, 'Dothan Celeron M'),
    'Intel Celeron N2806': ('DDR3L', 2014, 'Bay Trail'),
    'Intel Celeron N2830': ('DDR3L', 2014, 'Bay Trail'),
    'Intel Celeron N2840': ('DDR3L', 2014, 'Bay Trail'),
    'Intel Celeron N2940': ('DDR3L', 2014, 'Bay Trail'),
    'Intel Celeron N3000': ('DDR3L', 2015, 'Braswell'),
    'Intel Celeron N3050': ('DDR3L', 2015, 'Braswell'),
    'Intel Celeron N3060': ('DDR3L', 2016, 'Braswell'),
    'Intel Celeron N3150': ('DDR3L', 2015, 'Braswell'),
    'Intel Celeron N3160': ('DDR3L', 2016, 'Braswell'),
    'Intel Celeron N3350': ('DDR3L', 2016, 'Apollo Lake'),
    'Intel Celeron N3450': ('DDR3L', 2016, 'Apollo Lake'),
    'Intel Celeron N4000': ('DDR4', 2017, 'Gemini Lake'),
    'Intel Celeron N4020': ('DDR4', 2019, 'Gemini Lake Refresh'),
    'Intel Celeron N4100': ('DDR4', 2017, 'Gemini Lake'),
    'Intel Celeron N4120': ('DDR4', 2019, 'Gemini Lake Refresh'),
    'Intel Celeron N5100': ('DDR4', 2021, 'Jasper Lake'),
    
    # Intel Core
    'Intel Core 5 210H': ('DDR5', 2024, 'Meteor Lake Core 5'),
    'Intel Core 5 220U': ('DDR5', 2024, 'Meteor Lake Core 5'),
    'Intel Core Duo L2400': ('DDR2', 2006, 'Yonah low voltage'),
    'Intel Core Duo T2250': ('DDR2', 2006, 'Yonah'),
    'Intel Core Duo T2450': ('DDR2', 2006, 'Yonah'),
    'Intel Core Ultra 7 255U': ('DDR5', 2024, 'Meteor Lake Ultra'),
    'Intel Core i3-12100TE': ('DDR4', 2022, 'Alder Lake desktop'),
    'Intel Core i3-2357M': ('DDR3', 2011, 'Sandy Bridge ULV'),
    'Intel Core i3-9100': ('DDR4', 2019, 'Coffee Lake desktop'),
    'Intel Core i3-N305': ('DDR5', 2023, 'Alder Lake-N'),
    'Intel Core i5-1235U': ('DDR4', 2022, 'Alder Lake U'),
    'Intel Core i5-1245U': ('DDR4', 2022, 'Alder Lake U'),
    'Intel Core i5-12500TE': ('DDR4', 2022, 'Alder Lake desktop'),
    'Intel Core i5-12600H': ('DDR4', 2022, 'Alder Lake H'),
    'Intel Core i5-14500HX': ('DDR5', 2024, 'Raptor Lake Refresh'),
    'Intel Core i5-2410M': ('DDR3', 2011, 'Sandy Bridge mobile'),
    'Intel Core i5-3380M': ('DDR3', 2012, 'Ivy Bridge mobile'),
    'Intel Core i5-4308U': ('DDR3', 2013, 'Haswell ULT'),
    'Intel Core i5-430UM': ('DDR3', 2010, 'Arrandale ULV'),
    'Intel Core i5-450M': ('DDR3', 2010, 'Arrandale'),
    'Intel Core i7-12650HX': ('DDR5', 2022, 'Alder Lake HX'),
    'Intel Core i7-1265U': ('DDR4', 2022, 'Alder Lake U'),
    'Intel Core i7-1265UE': ('DDR4', 2022, 'Alder Lake embedded'),
    'Intel Core i7-12700TE': ('DDR4', 2022, 'Alder Lake desktop'),
    'Intel Core i7-4610M': ('DDR3', 2014, 'Haswell mobile'),
    'Intel Core i7-4712MQ': ('DDR3', 2014, 'Haswell quad mobile'),
    'Intel Core i7-4940MX': ('DDR3', 2014, 'Haswell Extreme mobile'),
    'Intel Core i7-5850HQ': ('DDR3', 2015, 'Broadwell HQ'),
    'Intel Core i7-610E': ('DDR3', 2010, 'Arrandale embedded'),
    'Intel Core i7-640LM': ('DDR3', 2010, 'Arrandale low voltage'),
    'Intel Core i7-7Y75': ('DDR3', 2016, 'Kaby Lake Y'),
    'Intel Core i7-940XM': ('DDR3', 2010, 'Clarksfield Extreme'),
    'Intel Core m3-8100Y': ('LPDDR3', 2018, 'Amber Lake Y'),
    'Intel Core m5-6Y54': ('DDR3L', 2015, 'Skylake Y'),
    'Intel Core m7-6Y75': ('DDR3L', 2015, 'Skylake Y'),
    'Intel Core2 Duo T5870': ('DDR2', 2008, 'Merom'),
    'Intel Core2 Duo T7700': ('DDR2', 2007, 'Merom'),
    
    # Intel N-series
    'Intel N100': ('DDR4', 2023, 'Alder Lake-N'),
    'Intel N200': ('DDR5', 2023, 'Alder Lake-N'),
    
    # Intel Pentium
    'Intel Pentium 4410Y': ('DDR3L', 2015, 'Skylake Y Pentium'),
    'Intel Pentium Gold 8505': ('DDR5', 2023, 'Raptor Lake Pentium'),
    'Intel Pentium M 1.20GHz': ('DDR', 2003, 'Banias Pentium M'),
    'Intel Pentium M 1000MHz': ('DDR', 2003, 'Banias Pentium M'),
    'Intel Pentium M 1400MHz': ('DDR', 2003, 'Banias Pentium M'),
    'Intel Pentium M 1500MHz': ('DDR', 2004, 'Dothan Pentium M'),
    'Intel Pentium T3200': ('DDR2', 2008, 'Penryn Pentium'),
    
    # Intel Xeon
    'Intel Xeon E3-1268L v5': ('DDR4', 2015, 'Skylake Xeon mobile'),
    'Intel Xeon E3-1505L v6': ('DDR4', 2017, 'Kaby Lake Xeon mobile'),
    'Intel Xeon E3-1535M v5': ('DDR4', 2015, 'Skylake Xeon mobile'),
    'Intel Xeon W-10855M': ('DDR4', 2020, 'Comet Lake Xeon W'),
    'Intel Xeon W-11855M': ('DDR4', 2021, 'Tiger Lake Xeon W'),
    'Intel Xeon W-11955M': ('DDR4', 2021, 'Tiger Lake Xeon W'),
    
    # Qualcomm
    'Qualcomm Snapdragon X Elite - X1E-78-100': ('LPDDR5X', 2024, 'Snapdragon X Elite'),
    'Qualcomm Snapdragon X Elite - X1E-80-100': ('LPDDR5X', 2024, 'Snapdragon X Elite'),
    'Snapdragon 835': ('LPDDR4X', 2017, 'Snapdragon mobile'),
    'Snapdragon X - X126100 - Qualcomm Oryon': ('LPDDR5X', 2024, 'Snapdragon X'),
    
    # VIA
    'VIA Nano U2250 (1.6GHz Capable)': ('DDR2', 2008, 'VIA Nano ULV'),
}

MISSING_CPUS_STORAGE = {
    # Format: cpu_name: (storage_type, storage_size, tier, notes)
    
    # AMD Ryzen (2024 = SSD 512GB+, 2020-2023 = SSD 512GB, older = SSD 256GB/HDD 500GB)
    'AMD Ryzen 5 2600H': ('SSD', '512GB', 'mid', '2018 Zen+ mobile H'),
    'AMD Ryzen 5 40': ('SSD', '512GB', 'mid', '2024 Zen 4'),
    'AMD Ryzen 5 7235HS': ('SSD', '512GB', 'mid', '2023 Zen 3+ Rembrandt'),
    'AMD Ryzen 5 8540U': ('SSD', '512GB', 'mid', '2024 Zen 4 Hawk Point'),
    'AMD Ryzen 5 8645HS': ('SSD', '512GB', 'mid', '2024 Zen 4 Hawk Point'),
    'AMD Ryzen 7 7435HS': ('SSD', '512GB', 'mid', '2023 Zen 3+ Rembrandt'),
    'AMD Ryzen 7 8845H': ('SSD', '512GB', 'mid', '2024 Zen 4 Hawk Point'),
    'AMD Ryzen 7 8845HS': ('SSD', '512GB', 'mid', '2024 Zen 4 Hawk Point'),
    'AMD Ryzen 9 7940HX': ('SSD', '1TB', 'high', '2023 Zen 4 Dragon Range'),
    'AMD Ryzen 9 8945HS': ('SSD', '512GB', 'high', '2024 Zen 4 Hawk Point'),
    
    # Apple Silicon (always SSD, M1/M2 = 256GB base, M3/M4 = 512GB base)
    'Apple M1 8 Core 3200 MHz': ('SSD', '256GB', 'mid', '2020 M1 base'),
    'Apple M1 Max 10 Core 3200 MHz': ('SSD', '512GB', 'high', '2021 M1 Max'),
    'Apple M1 Pro 10 Core 3200 MHz': ('SSD', '512GB', 'high', '2021 M1 Pro'),
    'Apple M2 8 Core 3500 MHz': ('SSD', '256GB', 'mid', '2022 M2 base'),
    'Apple M2 Pro 10 Core 3480 MHz': ('SSD', '512GB', 'high', '2023 M2 Pro'),
    'Apple M3 8 Core': ('SSD', '512GB', 'mid', '2023 M3 base'),
    'Apple M3 Max 14 Core': ('SSD', '512GB', 'high', '2023 M3 Max'),
    'Apple M3 Pro 11 Core': ('SSD', '512GB', 'high', '2023 M3 Pro'),
    'Apple M4 10 Core': ('SSD', '512GB', 'mid', '2024 M4 base'),
    
    # Intel Celeron (old = HDD, 2015+ = SSD 256GB, 2020+ = SSD 512GB)
    'Celeron Dual-Core T3000': ('HDD', '320GB', 'budget', '2008 Merom'),
    'Intel Atom C2338': ('SSD', '256GB', 'budget', '2013 Avoton'),
    'Intel Celeron 2955U': ('HDD', '500GB', 'budget', '2013 Haswell'),
    'Intel Celeron 3867U': ('SSD', '256GB', 'budget', '2017 Kaby Lake'),
    'Intel Celeron 7305': ('SSD', '256GB', 'budget', '2022 Alder Lake'),
    'Intel Celeron M 900MHz': ('HDD', '80GB', 'budget', '2004 Dothan'),
    'Intel Celeron N2840': ('HDD', '500GB', 'budget', '2014 Bay Trail'),
    'Intel Celeron N3050': ('SSD', '128GB', 'budget', '2015 Braswell'),
    'Intel Celeron N3060': ('SSD', '128GB', 'budget', '2016 Braswell'),
    'Intel Celeron N3350': ('SSD', '128GB', 'budget', '2016 Apollo Lake'),
    'Intel Celeron N4000': ('SSD', '128GB', 'budget', '2017 Gemini Lake'),
    'Intel Celeron N4020': ('SSD', '256GB', 'budget', '2019 Gemini Lake'),
    'Intel Celeron N4120': ('SSD', '256GB', 'budget', '2019 Gemini Lake'),
    
    # Intel Core (2020+ = SSD 512GB, 2015-2019 = SSD 256GB, older = HDD)
    'Intel Core 5 210H': ('SSD', '512GB', 'mid', '2024 Meteor Lake'),
    'Intel Core 5 220U': ('SSD', '512GB', 'mid', '2024 Meteor Lake'),
    'Intel Core Duo T2450': ('HDD', '160GB', 'budget', '2006 Yonah'),
    'Intel Core i3-12100TE': ('SSD', '512GB', 'budget', '2022 Alder Lake'),
    'Intel Core i5-1235U': ('SSD', '512GB', 'mid', '2022 Alder Lake'),
    'Intel Core i5-1245U': ('SSD', '512GB', 'mid', '2022 Alder Lake'),
    'Intel Core i5-12500TE': ('SSD', '512GB', 'mid', '2022 Alder Lake'),
    'Intel Core i5-2410M': ('HDD', '500GB', 'mid', '2011 Sandy Bridge'),
    'Intel Core i7-1265U': ('SSD', '512GB', 'mid', '2022 Alder Lake'),
    'Intel Core i7-610E': ('HDD', '320GB', 'mid', '2010 Arrandale'),
    'Intel Core m3-8100Y': ('SSD', '256GB', 'mid', '2018 Amber Lake'),
    'Intel Core m5-6Y54': ('SSD', '256GB', 'mid', '2015 Skylake Y'),
    
    # Intel Pentium
    'Intel Pentium Gold 8505': ('SSD', '256GB', 'budget', '2023 Raptor Lake'),
    
    # Intel Xeon (workstation = SSD 512GB+)
    'Intel Xeon W-11855M': ('SSD', '512GB', 'high', '2021 Tiger Lake W'),
    
    # Qualcomm
    'Qualcomm Snapdragon X Elite - X1E-78-100': ('SSD', '512GB', 'high', '2024 Snapdragon X'),
}


def append_to_ddr_map(filepath, cpus_dict):
    """Append missing CPUs to DDR map file."""
    print(f"\n📋 Adding {len(cpus_dict)} CPUs to {filepath}...")
    
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        count = 0
        for cpu_name, (ddr_type, year, notes) in sorted(cpus_dict.items()):
            writer.writerow([cpu_name, ddr_type, year, notes])
            count += 1
            if count <= 10 or count % 20 == 0:
                print(f"  ✓ Added: {cpu_name} → {ddr_type}")
    
    print(f"✅ Successfully added {count} CPUs to DDR map!")


def append_to_storage_map(filepath, cpus_dict):
    """Append missing CPUs to Storage map file."""
    print(f"\n📁 Adding {len(cpus_dict)} CPUs to {filepath}...")
    
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        count = 0
        for cpu_name, (storage_type, storage_size, tier, notes) in sorted(cpus_dict.items()):
            writer.writerow([cpu_name, storage_type, storage_size, tier, notes])
            count += 1
            if count <= 10 or count % 10 == 0:
                print(f"  ✓ Added: {cpu_name} → {storage_type} {storage_size}")
    
    print(f"✅ Successfully added {count} CPUs to Storage map!")


def main():
    base_dir = Path(__file__).parent
    ddr_map_file = base_dir / 'cpu_ddr_map.csv'
    storage_map_file = base_dir / 'cpu_storage_map.csv'
    
    print("=" * 70)
    print("ADD MISSING CPUs TO MAPPING FILES")
    print("=" * 70)
    
    # Add to DDR map
    if ddr_map_file.exists():
        append_to_ddr_map(ddr_map_file, MISSING_CPUS_DDR)
    else:
        print(f"❌ ERROR: {ddr_map_file} not found!")
    
    # Add to Storage map
    if storage_map_file.exists():
        append_to_storage_map(storage_map_file, MISSING_CPUS_STORAGE)
    else:
        print(f"❌ ERROR: {storage_map_file} not found!")
    
    print("\n" + "=" * 70)
    print("✅ DONE! Missing CPUs have been added to the mapping files.")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Review the added entries in the CSV files")
    print("   2. Re-run clean_ram_storage.ipynb")
    print("   3. Verify that 'CPU not in map' counts are now much lower")
    print("=" * 70)


if __name__ == '__main__':
    main()
