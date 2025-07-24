import os

# Directory containing the NIfTI maps
maps_dir = '/data/liuq13/SPCD_decoding_subset/searchlight_maps'
swarm_file = 'smooth.swarm'
script_name = 'smooth_one.py'

lines = []

# Walk through all files and select *.nii.gz (excluding already smoothed ones)
for root, _, files in os.walk(maps_dir):
    for f in files:
        if f.endswith('.nii.gz') and 'smooth' not in f:
            full_path = os.path.join(root, f)
            cmd = f"python3 {script_name} {full_path}"
            lines.append(cmd)

# Write to swarm file
with open(swarm_file, 'w') as f:
    f.write('\n'.join(lines))

print(f"✅ Wrote {len(lines)} lines to {swarm_file}")
