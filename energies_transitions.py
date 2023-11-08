## List of transitions for Er and Dy ##

from math import pi

Wls = {'401': 401e-9,
      '583': 583e-9,
      '631': 631e-9,
      '626': 626e-9,
      '841': 841e-9,
      '488': 488e-9,
      '486': 486e-9}

Gammas = {'401': 2*pi*28e6,
          '583': 2*pi*180e3,
          '631': 2*pi*28e3,
          '626': 2*pi*135e3,
          '841': 2*pi*8e3,
          '488': 0,
          '486': 0}

conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
alpha_GS = 430 * conversion # polarizability
