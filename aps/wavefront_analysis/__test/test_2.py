
propagation_distance = -0.994

######################################################
# delta theta +20 urad : Shadow ~-60 um

ref_speckle_shift  = [-1.1, 1.4]
speckle_shift      = [-1.1, 28.0]

pixel_size = 3.35e-01 * 2.0

speckle_shift_x  = pixel_size * (speckle_shift[1]  - ref_speckle_shift[1])
speckle_shift_y  = pixel_size * (speckle_shift[0]  - ref_speckle_shift[0])

def calculate_shift(speckle_shift, propagation_distance):
    return -speckle_shift *(abs(propagation_distance) - 0.2) / 0.2

print("shift", round(calculate_shift(speckle_shift_x, propagation_distance), 2),"um")


######################################################
# delta T +50 um : Shadow ~-32 um

ref_speckle_shift  = [-8.0, 8.2]
speckle_shift      = [-7.9, 25.0]

pixel_size = 3.35e-01 * 2.0

speckle_shift_x  = pixel_size * (speckle_shift[1]  - ref_speckle_shift[1])
speckle_shift_y  = pixel_size * (speckle_shift[0]  - ref_speckle_shift[0])

def calculate_shift(speckle_shift, propagation_distance):
    return -speckle_shift *(abs(propagation_distance) - 0.2) / 0.2

print("shift", round(calculate_shift(speckle_shift_x, propagation_distance), 2),"um")
