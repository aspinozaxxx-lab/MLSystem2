from urban.functional.regression import crop_by_mask
from urban.functional.io import read_bc, read_fc
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

bc = read_bc('tests/test_data/real/Ufa', ('RED', 'GRN', 'BLU'))
fc = read_fc('tests/test_data/real/Ufa', '100').to_crs(bc.crs)

crops = []
_, ax = plt.subplots(nrows=3, ncols=3)
for i in range(9):
    geom = fc[i+100, 'geometry']
    ax[i//3, i%3].imshow(crop_by_mask(bc, geom.buffer(4), (512,512)).transpose(1, 2, 0))


plt.show()