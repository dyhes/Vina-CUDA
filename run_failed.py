import os
import subprocess
import numpy as np

def get_mol2_center(filepath):
    coords = []
    try:
        with open(filepath, 'r') as f:
            reading = False
            for line in f:
                if line.startswith('@<TRIPOS>ATOM'):
                    reading = True
                    continue
                if line.startswith('@<TRIPOS>BOND'):
                    break
                if reading:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            coords.append((x, y, z))
                        except ValueError:
                            pass
        if not coords: return 0, 0, 0
        coords = np.array(coords)
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c) / 2
        return center[0], center[1], center[2]
    except:
        return 0, 0, 0

failed = ['1g2k', '1q8u', '2iwx', '2vw5', '2w66', '2wer', '2x00', '2xys', '2yge', '3gnw', '3u8k', '3u8n', '3utu', '5c2h']
coreset_dir = "/home/ubuntu/grad/dataset/CASF-2016-PDBQT/coreset"
mol2_dir = "/home/ubuntu/grad/dataset/CASF-2016/coreset"
binary = "/home/ubuntu/grad/variants/QuickVina-W-CUDA/Vina-GPU-2-1-W-CUDA"
out_dir = "/home/ubuntu/grad/results_qvw"

for target in failed:
    target_dir = os.path.join(coreset_dir, target)
    rec_pdbqt = os.path.join(target_dir, f"{target}_protein.pdbqt")
    lig_pdbqt = os.path.join(target_dir, f"{target}_ligand.pdbqt")
    original_ligand = os.path.join(mol2_dir, target, f"{target}_ligand.mol2")
    
    cx, cy, cz = get_mol2_center(original_ligand)
    sx, sy, sz = 22.5, 22.5, 22.5
    out_file = os.path.join(out_dir, f"{target}_out.pdbqt")
    
    cmd = [
        "sudo", "bash", "-c",
        f"ulimit -s 8192 && {binary} --receptor {rec_pdbqt} --ligand {lig_pdbqt} "
        f"--center_x {cx:.3f} --center_y {cy:.3f} --center_z {cz:.3f} "
        f"--size_x {sx:.3f} --size_y {sy:.3f} --size_z {sz:.3f} "
        f"--thread 8192 --out {out_file}"
    ]
    print(f"Running {target}...")
    subprocess.run(" ".join(cmd), shell=True)

