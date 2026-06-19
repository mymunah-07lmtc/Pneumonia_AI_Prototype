import pandas as pd

# The OFFICIAL Healthy patients from the ICBHI 2017 dataset (26 patients)
# All other patients (100) have some respiratory disease (COPD, URTI, Asthma, Pneumonia, etc.)
healthy_patients = [
    103, 113, 117, 120, 122, 123, 124, 127, 128, 129, 131, 132, 133,
    134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147
]

# Generate all patients from 101 to 226
all_patients = list(range(101, 227))

data = []
for pid in all_patients:
    if pid in healthy_patients:
        data.append([pid, 'Healthy'])
    else:
        data.append([pid, 'Abnormal'])

df = pd.DataFrame(data, columns=['Patient_ID', 'Diagnosis'])

# Save and overwrite the bad file
df.to_csv("icbhi_diagnosis.csv", index=False)

print("✅ CORRECT mapping saved to icbhi_diagnosis.csv")
print(f"   Total patients: {len(df)}")
print(f"   Healthy: {len(healthy_patients)}")
print(f"   Abnormal: {len(df) - len(healthy_patients)}")
print(f"   Healthy Patient IDs: {healthy_patients}")