import numpy as np


def load_pcd(path):
    """Read an ASCII PCD with fields x y z rgb (rgb packed as uint32, as PCL writes it).
    Returns xyz (N,3) float64 and rgb (N,3) float64 in 0..1."""
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode().strip()
            if not line or line.startswith('#'):
                continue
            key, *vals = line.split()
            header[key] = vals
            if key == 'DATA':
                break
        if header['DATA'][0] != 'ascii':
            raise ValueError(f"{path}: only ASCII PCD is supported")
        data = np.loadtxt(f, dtype=np.float64, ndmin=2)
    fields = header['FIELDS']
    xyz = data[:, [fields.index(c) for c in 'xyz']]
    packed = data[:, fields.index('rgb')].astype(np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1) / 255.0
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok], rgb[ok]
