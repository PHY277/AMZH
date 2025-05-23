import math
#function for turning arc mins into degrees
def degrees_arcmins (degrees,arcmins):
    tot_degrees = degrees + (arcmins/60)
    return tot_degrees

#function for Average degree
def find_avg_degree (L,R):
    avg_degree = (L+R)/2
    return avg_degree
    
#function for finding wavelength
def find_wavelength (degree):
    a = 1.66*10**3
    radians = math.radians(degree)
    wavelength = (a*math.sin(radians))
    return wavelength

#combining all the functions
def final_function(L_deg,L_min,R_deg,R_min,L_slit_deg=55,L_slit_min=8,R_slit_deg=55,R_slit_min=10):
    
    #converting to totally degrees
    L_tot_degrees = degrees_arcmins (L_deg,L_min)
    R_tot_degrees = degrees_arcmins (R_deg,R_min)
    L_slit_tot_degrees = degrees_arcmins (L_slit_deg,L_slit_min)
    R_slit_tot_degrees = degrees_arcmins (R_slit_deg,R_slit_min)
    
    #Finding the average and taking slit degree into account
    avg_degree = find_avg_degree (L_tot_degrees,R_tot_degrees)
    avg_slit_deg = find_avg_degree (L_slit_tot_degrees,R_slit_tot_degrees)
    final_deg = abs(avg_degree-avg_slit_deg)
    
    #finding the wavelength 
    wavelength = find_wavelength (final_deg)
    
    #Rounding
    final_deg = round(final_deg,2)
    wavelength = round(wavelength,2)
    
    return final_deg,wavelength

print('\n(Color, line number): (degree (°),wavelength (nm))')
print('\nHydrogen:')
print('\n Left')
print('Red:',final_function(32,7,32,8))
print('cyan/green:',final_function(38,9,38,10))
print('Blue:',final_function(40,8,40,9))
print('\n Right')
print('Red:',final_function(78,0,78,1))
print('cyan/green:',final_function(72,8,72,9))
print('Blue:',final_function(70,9,70,10))

print('\n\nHelium')
print('\n Left')
print('Red:',final_function(30,9,30,8))
print('Red:',final_function(31,4,31,3))
print('Yellow:',final_function(34,4,34,3))
print('cyan/green:',final_function(37,5,37,4))
print('cyan/green:',final_function(37,8,37,7))
print('Blue:',final_function(39,6,39,5))
print('Blue:',final_function(39,4,39,5))

print('\n\nUnknown C')
print('Violet 1:', final_function(69,2,69,3))
print('Violet 2:', final_function(69,5,69,6))
print('Violet 3:', final_function(69,7,69,8))
print('Violet 4:', final_function(70,0,70,1))
print('Blue/Purple 5:', final_function(70,5,70,6))
print('Blue 6:', final_function(70,7,70,9))
print('Blue/green 7:', final_function(71,4,71,5))
print('Green 8:', final_function(74,0,74,1))
print('Yellow 9:', final_function(75,0,75,1))
print('Yellow 10:', final_function(75,3,75,4))
print('Orange 11:', final_function(75,8,75,9))
print('Red 12:', final_function(77,1,77,3))
print('Red 13:', final_function(77,6,77,8))
print('Red 14:', final_function(78,2,78,4))

print('\n\nUnknown B')
print('Green 1:', final_function(36,4,36,3))
print('Green 2:', final_function(36,2,36,0))
print('Green 3:', final_function(34,8,34,7))
print('Yellow 4:', final_function(34,5,34,4))
print('Yellow 5:', final_function(34,4,34,3))
print('Orange/Yellow 6:', final_function(34,1,34,0))
print('Orange/Yellow 7:', final_function(34,0,33,9))
print('Orange 8:', final_function(33,8,33,7))
print('Orange 9:', final_function(33,7,33,5))
print('Orange 10:', final_function(33,4,33,3))
print('Red 11:', final_function(33,1,33,0))
print('Red 12:', final_function(32,9,32,8))
print('Red 13:', final_function(32,8,32,7))
print('Red 14:', final_function(32,7,32,6))
print('Red 15:', final_function(32,5,32,4))
print('Red 16:', final_function(32,4,32,3))
print('Red 17:', final_function(32,0,31,9))
print('Red 18:', final_function(31,9,31,8))
print('Red 19:', final_function(31,7,31,6))
print('Red 20:', final_function(31,4,31,3))
print('Red 21:', final_function(31,3,31,1))