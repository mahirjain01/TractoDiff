import nibabel as nib

tractogram_file = '/tracto/Gagan/data/TractoInferno/derivatives/trainset/sub-1030/tractography/sub-1030__AF_L.trk'
tractogram = nib.streamlines.load(tractogram_file)
streamlines = tractogram.streamlines

print(f"Number of streamlines: {len(streamlines)}\n")

# Show first 3 streamlines
for idx, streamline in enumerate(streamlines[:3]):
    print(f"Streamline {idx + 1} (length: {len(streamline)} points):")
    print(streamline)  # streamline is a (N, 3) array of XYZ coordinates
    print()
