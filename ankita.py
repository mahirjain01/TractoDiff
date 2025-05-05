#%%

import nibabel as nib

trk_file = '/tracto/TractoDiff/data/testset/sub-1160/tractography/sub-1160__AF_L.trk'
# '/tracto/TractoDiff/visualizations/generated.trk'

strls = nib.streamlines.load(trk_file)

print(strls)
# print(strls.streamlines)
print(type(strls.streamlines))
print(len(strls.streamlines))

# sub-1119, 1159, 1160, 1030, 1079; AF_L

