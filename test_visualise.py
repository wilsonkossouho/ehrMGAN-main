# test_visualise.py
import numpy as np
from visualise import visualise_vae, visualise_gan

# Données factices
continuous_real = np.random.rand(5, 24, 7)
continuous_fake = np.random.rand(5, 24, 7)
discrete_real = np.random.randint(0, 2, (5, 24, 3))
discrete_fake = np.random.randint(0, 2, (5, 24, 3))

# Tester
visualise_vae(continuous_real, continuous_fake, discrete_real, discrete_fake, inx=999)
visualise_gan(continuous_real, continuous_fake, discrete_real, discrete_fake, inx=999)

print("✅ Si des PNG sont créés dans logs/visualizations/, tout fonctionne !")