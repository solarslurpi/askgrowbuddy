# Playing around with the Saturated Paste data looking at the cation balance.
# Corrected data from soil test
Ca_ppm = 356.90  # From saturated paste test
Mg_ppm = 26.99   # From saturated paste test
K_ppm = 88.13    # From saturated paste test
Na_ppm = 11.41   # Corrected value from saturated paste test

# Convert ppm to meq/L
#1 ppm means 1 part per million parts, regardless of what those parts are or how they're measured.
# meq tells us how much electrical charge the ppm provides.
#  (40.08 g/mol)/2 eq/mol = 20.04 g/eq
# 1 meq = 1/1000 eq
# 1 meq/L = 20.04 mg Ca2+
# 1/20.04 meq/L per ppm = 0.0499 = 1 mg Ca2+
# Conversion factors
Ca_factor = 0.0499
# Mg = 24.305 g/mol. There are 2+ charge. equivalent weight = 24.305/2 = 12.1525 mg = 1 meq of Mg. Converstion factor = 1/12.1525 = 0.08229 meq/L per ppm
Mg_factor = 0.08229
K_factor = 0.02557
Na_factor = 0.04350

Ca_meq = Ca_ppm * Ca_factor
Mg_meq = Mg_ppm * Mg_factor
K_meq = K_ppm * K_factor
Na_meq = Na_ppm * Na_factor

total_meq = Ca_meq + Mg_meq + K_meq + Na_meq

# Calculate percentages
Ca_percent = (Ca_meq / total_meq) * 100
Mg_percent = (Mg_meq / total_meq) * 100
K_percent = (K_meq / total_meq) * 100
Na_percent = (Na_meq / total_meq) * 100

print(f"Calcium: {Ca_percent:.2f}%")
print(f"Magnesium: {Mg_percent:.2f}%")
print(f"Potassium: {K_percent:.2f}%")
print(f"Sodium: {Na_percent:.2f}%")

# Calculate ratios
Ca_Mg_ratio = Ca_meq / Mg_meq
Ca_K_ratio = Ca_meq / K_meq

print(f"\nCa:Mg ratio: {Ca_Mg_ratio:.2f}")
print(f"Ca:K ratio: {Ca_K_ratio:.2f}")