import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# function for turning arc mins into degrees
def degrees_arcmins(degrees, arcmins):
    tot_degrees = degrees + (arcmins / 60)
    return tot_degrees

# function for Average degree
def find_avg_degree(L, R):
    avg_degree = (L + R) / 2
    return avg_degree

# function for finding wavelength
def find_wavelength(degree):
    a = 1.66 * 10**3
    radians = math.radians(degree)
    wavelength = a * math.sin(radians)
    return wavelength

# function for finding error in wavelength
def find_wavelength_error(degree, angle_error_deg):
    a = 1.66 * 10**3
    radians = math.radians(degree)
    angle_error_rad = math.radians(angle_error_deg)
    wavelength_error = a * math.cos(radians) * angle_error_rad
    return wavelength_error

# combining all the functions
def final_function(L_deg, L_min, R_deg, R_min,
                   L_slit_deg=55, L_slit_min=8, R_slit_deg=55, R_slit_min=10,
                   return_errors=False):

    # converting to total degrees
    L_tot_degrees = degrees_arcmins(L_deg, L_min)
    R_tot_degrees = degrees_arcmins(R_deg, R_min)
    L_slit_tot_degrees = degrees_arcmins(L_slit_deg, L_slit_min)
    R_slit_tot_degrees = degrees_arcmins(R_slit_deg, R_slit_min)

    # Finding the average and taking slit degree into account
    avg_degree = find_avg_degree(L_tot_degrees, R_tot_degrees)
    avg_slit_deg = find_avg_degree(L_slit_tot_degrees, R_slit_tot_degrees)
    final_deg = abs(avg_degree - avg_slit_deg)

    # finding the wavelength 
    wavelength = find_wavelength(final_deg)

    # error propagation
    angle_error_deg = 0.5 / 60  # 0.5 arcmin in degrees
    wavelength_error = find_wavelength_error(final_deg, angle_error_deg)

    # Rounding
    final_deg = round(final_deg, 2)
    wavelength = round(wavelength, 2)
    angle_error_deg = round(angle_error_deg, 5)
    wavelength_error = round(wavelength_error, 2)

    if return_errors:
        return final_deg, wavelength, angle_error_deg, wavelength_error
    else:
        return final_deg, wavelength  

# Setup n_low and n_up values (Balmer series)
n_low = 2
n_up_values = np.array([3, 4, 5])  # Red, Cyan/Green, Blue

# Compute x-axis values (1/n_low^2 - 1/n_up^2)
x_values = 1 / n_low**2 - 1 / n_up_values**2

# Your Rotate Left measured wavelengths (in nm) and errors
wavelengths_left = np.array([643.87, 485.35, 434.77])  # nm
errors_left = np.array([0.62, 0.62, 0.62])  # nm

# Your Rotate Right measured wavelengths (in nm) and errors
wavelengths_right = np.array([644.08, 485.57, 435.02])  # nm
errors_right = np.array([0.62, 0.62, 0.62])  # nm

# Convert wavelengths to 1/m and calculate errors
def wavelength_to_inverse(wavelength_nm, error_nm):
    wavelength_m = wavelength_nm * 1e-9
    inverse_lambda = 1 / wavelength_m
    inverse_lambda_err = error_nm * 1e-9 / wavelength_m**2
    return inverse_lambda, inverse_lambda_err

inv_left, inv_left_err = wavelength_to_inverse(wavelengths_left, errors_left)
inv_right, inv_right_err = wavelength_to_inverse(wavelengths_right, errors_right)

#  Define a linear model for curve fitting
def linear(x, R):
    return R * x

# Fit and plot for Rotate Left
popt_left, pcov_left = curve_fit(linear, x_values, inv_left, sigma=inv_left_err, absolute_sigma=True)
R_left = popt_left[0]
R_left_err = np.sqrt(np.diag(pcov_left))[0]

plt.errorbar(x_values, inv_left, yerr=inv_left_err, fmt='o', label='Rotate Left data', color='blue')
plt.plot(x_values, linear(x_values, *popt_left), '-', label=f'Fit: R = ({R_left:.2e} ± {R_left_err:.2e}) m⁻¹', color='navy')
plt.xlabel(r'$1/n_{low}^2 - 1/n_{up}^2$')
plt.ylabel(r'$1/\lambda$ (m⁻¹)')
plt.title('Hydrogen Rotate Left: Rydberg Constant')
plt.legend()
plt.grid(True)
plt.show()

# Fit and plot for Rotate Right
popt_right, pcov_right = curve_fit(linear, x_values, inv_right, sigma=inv_right_err, absolute_sigma=True)
R_right = popt_right[0]
R_right_err = np.sqrt(np.diag(pcov_right))[0]

plt.errorbar(x_values, inv_right, yerr=inv_right_err, fmt='o', label='Rotate Right data', color='green')
plt.plot(x_values, linear(x_values, *popt_right), '-', label=f'Fit: R = ({R_right:.2e} ± {R_right_err:.2e}) m⁻¹', color='darkgreen')
plt.xlabel(r'$1/n_{low}^2 - 1/n_{up}^2$')
plt.ylabel(r'$1/\lambda$ (m⁻¹)')
plt.title('Hydrogen Rotate Right: Rydberg Constant')
plt.legend()
plt.grid(True)
plt.show()

# Print out results
print(f"Rotate Left Experimental Rydberg Constant: ({R_left:.2e} ± {R_left_err:.2e}) m⁻¹")
print(f"Rotate Right Experimental Rydberg Constant: ({R_right:.2e} ± {R_right_err:.2e}) m⁻¹")

# Reference value for comparison
R_true = 1.097373e7  # m⁻¹
print(f"Accepted Rydberg Constant: {R_true:.2e} m⁻¹")

print('\n(Color, line number): (degree (°),wavelength (nm))')
print('\nHydrogen:')
print('\n Left')
print('Red:',final_function(32,7,32,8,return_errors=True))
print('cyan/green:',final_function(38,9,38,10,return_errors=True))
print('Blue:',final_function(40,8,40,9,return_errors=True))
print('\n Right')
print('Red:',final_function(78,0,78,1,return_errors=True))
print('cyan/green:',final_function(72,8,72,9,return_errors=True))
print('Blue:',final_function(70,9,70,10,return_errors=True))

print('\n\nHelium')
print('\n Left')
print('Red:',final_function(30,9,30,8,return_errors=True))
print('Red:',final_function(31,4,31,3,return_errors=True))
print('Yellow:',final_function(34,4,34,3,return_errors=True))
print('cyan/green:',final_function(37,5,37,4,return_errors=True))
print('cyan/green:',final_function(37,8,37,7,return_errors=True))
print('Blue:',final_function(39,6,39,5,return_errors=True))
print('Blue:',final_function(39,4,39,5,return_errors=True))

print('\n\nUnknown C')
print('Violet 1:', final_function(69,2,69,3,return_errors=True))
print('Violet 2:', final_function(69,5,69,6,return_errors=True))
print('Violet 3:', final_function(69,7,69,8,return_errors=True))
print('Violet 4:', final_function(70,0,70,1,return_errors=True))
print('Blue/Purple 5:', final_function(70,5,70,6,return_errors=True))
print('Blue 6:', final_function(70,7,70,9,return_errors=True))
print('Blue/green 7:', final_function(71,4,71,5,return_errors=True))
print('Green 8:', final_function(74,0,74,1,return_errors=True))
print('Yellow 9:', final_function(75,0,75,1,return_errors=True))
print('Yellow 10:', final_function(75,3,75,4,return_errors=True))
print('Orange 11:', final_function(75,8,75,9,return_errors=True))
print('Red 12:', final_function(77,1,77,3,return_errors=True))
print('Red 13:', final_function(77,6,77,8,return_errors=True))
print('Red 14:', final_function(78,2,78,4,return_errors=True))

print('\n\nUnknown B')
print('Green 1:', final_function(36,4,36,3,return_errors=True))
print('Green 2:', final_function(36,2,36,0,return_errors=True))
print('Green 3:', final_function(34,8,34,7,return_errors=True))
print('Yellow 4:', final_function(34,5,34,4,return_errors=True))
print('Yellow 5:', final_function(34,4,34,3,return_errors=True))
print('Orange/Yellow 6:', final_function(34,1,34,0,return_errors=True))
print('Orange/Yellow 7:', final_function(34,0,33,9,return_errors=True))
print('Orange 8:', final_function(33,8,33,7,return_errors=True))
print('Orange 9:', final_function(33,7,33,5,return_errors=True))
print('Orange 10:', final_function(33,4,33,3,return_errors=True))
print('Red 11:', final_function(33,1,33,0,return_errors=True))
print('Red 12:', final_function(32,9,32,8,return_errors=True))
print('Red 13:', final_function(32,8,32,7,return_errors=True))
print('Red 14:', final_function(32,7,32,6,return_errors=True))
print('Red 15:', final_function(32,5,32,4,return_errors=True))
print('Red 16:', final_function(32,4,32,3,return_errors=True))
print('Red 17:', final_function(32,0,31,9,return_errors=True))
print('Red 18:', final_function(31,9,31,8,return_errors=True))
print('Red 19:', final_function(31,7,31,6,return_errors=True))
print('Red 20:', final_function(31,4,31,3,return_errors=True))
print('Red 21:', final_function(31,3,31,1,return_errors=True))


# In[ ]:




