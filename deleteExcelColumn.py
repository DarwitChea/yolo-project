import pandas as pd

attendance_file = "attendance.xlsx"

# Load the file
df = pd.read_excel(attendance_file)

# Keep only the first 3 columns
df = df.iloc[:, :3]

# Save back to the same file (overwrites sessions)
df.to_excel(attendance_file, index=False)

print("✅ All session columns reset.")
