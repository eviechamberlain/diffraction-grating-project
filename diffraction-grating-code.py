#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import ticker
from numpy import ma
from scipy.special import j1
import os

#Define your functions that you need in your code here
filename = "C:/Users/py22ec/Downloads/assessment_data_py22ec.dat"
#function that makes the metadata dictionary
def makemeta(filename):
    """
creates a dictionary called metadata containing the information from the first part of the file

Parameters:
    filename (str): name of the data file

Raises:
    RuntimeError : if thhe line has more than one equals in it
       

Returns:
    metadata (dict): a dictionary containing the first part of the data, keys are the first part and defintions are the secdond part

    """
    metadata = {}
    with open(filename, "r") as fulldata:
        for line in fulldata:
            line = line.strip()
            if line == "&END":
                break
            if "=" in line:
                keydef = line.split("=")
                if len(keydef) != 2:
                    raise RuntimeError(f"{line} have too many equals signs in it so it cant go in the dictionary, metadata")
                metadata[keydef[0]] = keydef[1]
    return metadata

metadata = makemeta(filename)
#function which calculates k
k = (2*np.pi / float(metadata["Wavelength (nm)"]))*10e9

#chi squared
def chisqu(observed, fitted):
    """
performs a chi squared test to determine goodness of fit

Parameters:
    observed (array-like): Array containing the observed values.
    fitted (array-like): Array containing the expected values.

Returns:
    float: returns the calculated chi-squared value.

Raises:
    ValueError: if the observed and fitted arrays are not the same length
    """
    if len(observed) != len(fitted):
        raise ValueError("observed and fitted arrays must be the same length")
    return np.sum(((fitted - observed)**2)/(len(fitted)-3))

#create the squarefit
def squarefit(sinwhat, I0, w):
    """
uses the formula from the instructions to calculate the intensity distribution for a square or rectangle aperture using given parameters

Parameters:
    sinwhat (array-like): Array of sine values representing angles.
    I0 (float): Maximum intensity of the square or rectangle aperture.
    w (float): Width of the square or rectangle aperture.

Returns:
    array-like: Intensity values computed for the given sine values.

    """
    bruh =k*(w/2)*sinwhat
    Iwhat = I0*(np.sinc(bruh))**2
    return Iwhat

#create the diamondfit
def diafit(sinsth,  I0, w):
    """
uses the formula from the instructions to calculate the intensity distribution for a diamond aperture aperture using given parameters

Parameters:
    sinwhat (array-like): Array of sine values representing angles.
    I0 (float): Maximum intensity of the diamond aperture.
    w (float): Width of the diamond aperture.

Returns:
    array-like: Intensity values computed for the given sine values.

    """
    insd = k*w*sinsth / (2*np.sqrt(2))
    Iwhich = I0*np.sinc(insd)**4
    return Iwhich
   
#create the circlefit
def circfit(sinn, I0, D):
    """
uses the formula from the instructions to calculate the intensity distribution for a circle aperture using given parameters

Parameters:
    sinwhat (array-like): Array of sine values representing angles.
    I0 (float): Maximum intensity of the circle aperture.
    w (float): Width of the circle aperture.

Returns:
    array-like: Intensity values computed for the given sine values.

    """
    insidebessel = np.pi * k * D *sinn
    part111 = 2*j1(insidebessel)
    Iwho = I0 * (part111/ ( np.pi * k* D* sinn))**2
    # Masking to remove zeros
    Iwho = np.ma.masked_where(Iwho == 0, Iwho)
    return Iwho

def squ_error(num1, num2, error1, error2):
    """
takes 2 numbers with errors, and combines the errors by adding them in quadature

Parameters:
    num1 (float): number corresponding to first error
    num2 (float): number corresponding to second error
    error1 (float): error in the first number
    error2 (float): error in the second number
   
Returns:
    squ_error (float): the combination of errors in dimesnsion
    """
    squ_error_squared = (error1/num1)**2 + (error2/num2)**2    
    squ_error = np.sqrt(squ_error_squared)
    return squ_error

def ProcessData(filename):
    """
Processes data from a given file containing diffraction pattern measurements.

Parameters:
    filename (str): The path to the data file to be processed.

Returns:
    dict: A dictionary containing the results of the data processing, including:
        - 'shape' (str): The shape of the aperture, which can be one of: "square", "rectangle", "diamond", or "circle".
        - 'dim_1' (float): The first dimension of the aperture expressed in microns.
        - 'dim_1_err' (float): The uncertainty in the first dimension, also expressed in microns.
        - 'dim_2' (float): The second dimension of the aperture for rectangles, or the same as 'dim_1' for other shapes.
        - 'dim_2_err' (float): The uncertainty in the second dimension, also expressed in microns.
        - 'I0/area' (float): The fitted overall intensity value per unit area of the aperture.
        - 'I0/area_err' (float): The uncertainty in the overall intensity value per unit area.

Raises:
    FileNotFoundError: If the data file is not found.
    KeyError: If the metadata that is used in later calculations isn't in the metadata.
    RuntimeError: If there's an issue parsing metadata from the data file.

    """
    #Your ProcessData code goes here
    # Check if the data file exists
    data_file_path = filename
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found at: {data_file_path}")
       
    #call the function make metadata to define the metadata
        #create the dictionary "metadata"
       
   
    with open(filename, "r") as fulldata:
        for line in fulldata:
            line = line.strip()
            if line == "&END":
                data = np.loadtxt(fulldata)
                break
           
       
    metadata = makemeta(filename)              
    # Check if the wavelength is provided in metadata
    if "Wavelength (nm)" not in metadata:
        raise KeyError("Wavelength information not found in metadata")
                           
    # Check if Distance to Screen is provided in metadata
    if "Distance to Screen (m)" not in metadata:
        raise KeyError("Distance to Screen information not found in metadata")
                               
    #taking the first row and collunm of the data, not including the "nan"    
    ydats = data[1:, 0]
    xdats = data[0, 1:]
                               
    #account for the "as projected" option
    distfromscr = float(metadata["Distance to Screen (m)"])*1000
    sinphi = ydats / np.sqrt(ydats**2 + distfromscr**2)
    sintheta = xdats / np.sqrt(xdats**2 + distfromscr**2)
    smallerdata = data[1: , 1:]
    smallerdata[smallerdata < 0] = 0
    X, Y = np.meshgrid(sintheta, sinphi)
    fig, ax = plt.subplots()
    smallerdata = ma.masked_where(smallerdata <= 0, smallerdata)
    cs = ax.contourf(X, Y, smallerdata, locator=ticker.LogLocator())
    cbar = fig.colorbar(cs)
    plt.title("Diffraction pattern data for py22ec")
    plt.xlabel(r"sin $\theta$")
    plt.ylabel(r"sin $\phi$")
    plt.show()
   
    #getting the middle of the data for theta
    middleyloc = len(sintheta) / 2
    middleylocint = int(middleyloc)
    middleydat = smallerdata[middleylocint, : ]

    #getting the middle of the data for phi
    middlexloc = len(sinphi) / 2
    middlexlocint = int(middlexloc)
    middlexdat = smallerdata[ : , middlexlocint]


    #calculate k
    k = (2*np.pi / float(metadata["Wavelength (nm)"]))*10e9
   
   
    #fitting the square
    popt1s, pcov1s = curve_fit(squarefit, sinphi, middlexdat, p0=[np.max(middlexdat), 2e-5])

    popt2s, pcov2s = curve_fit(squarefit, sintheta, middleydat, p0=[np.max(middleydat), 2e-5])

    #fitting the circle
    popt1c, pcov1c = curve_fit(circfit, sinphi, middlexdat, p0=[np.max(middleydat), 2e-5])

    popt2c, pcov2c = curve_fit(circfit,sintheta,middleydat, p0=[np.max(middleydat), 2e-5])

    #fitting the diamond
    popt1d, pcov1d = curve_fit(diafit, sinphi, middlexdat, p0=[np.max(middleydat), 2e-5])

    popt2d, pcov2d = curve_fit(diafit, sintheta, middleydat, p0=[np.max(middleydat), 2e-5])


    #chi squared for each shape
    square_expected = squarefit(sinphi, *popt1s)
    chi_for_square = chisqu(middlexdat, square_expected)

    diag_expected = diafit(sinphi, *popt1d)
    chi_for_diag = chisqu(middlexdat, diag_expected)

    circ_expected = circfit(sinphi, *popt1c)
    chi_for_circ = chisqu(middlexdat, circ_expected)


    #determining if the shape is a circle or not

    if min(chi_for_diag,chi_for_square,chi_for_circ) == chi_for_circ:
        poptfin1, pcovfin1 = popt1c, pcov1c
        poptfin2, pcovfin2 = popt2c, pcov2c
        shape = "circle"
        """ #plot circle
        plt.plot(np.linspace(-0.05, 0.05, 1000),circfit(np.linspace(-0.05,0.05,1000), *popt1c))
        plt.plot(sinphi, middlexdat, "+")
        plt.title("Position Vs intensity for circular fit")
        plt.show()
       
        plt.plot(np.linspace(-0.05, 0.05, 1000),circfit(np.linspace(-0.05,0.05,1000), *popt2c))
        plt.plot(sintheta, middleydat, "+")
        plt.title("Position Vs intensity for circular fit 2")
        plt.show() """

    #determining if the shape is square/rectangle or diamond
    elif min(chi_for_diag,chi_for_square,chi_for_circ) == chi_for_square:
        poptfin1, pcovfin1 = popt1s, pcov1s
        poptfin2, pcovfin2 = popt2s, pcov2s
       
            #determining if the shape is a square or rectangle
        if abs(popt1s[1] - popt2s[1]) >= 0.05e-6:
            shape = "rectangle"
            """ #plot rectangle
            plt.plot(np.linspace(-0.05, 0.05, 1000), squarefit(np.linspace(-0.05,0.05,1000), *popt1d))
            plt.plot(sinphi, middlexdat, "+")
            plt.title("Position Vs. Intensity fit for horizontal slice")
            plt.show()
           
            plt.plot(np.linspace(-0.05, 0.05,1000), squarefit(np.linspace(-0.05,0.05,1000), *popt2s))
            plt.plot(sintheta, middleydat, "+")
            plt.title("Position Vs. Intensity for vertical slice")
            plt.show() """
        else:
            shape = "square"
            """ #plot square
            plt.plot(np.linspace(-0.05, 0.05, 1000), squarefit(np.linspace(-0.05,0.05,1000), *popt1d))
            plt.plot(sinphi, middlexdat, "+")
            plt.title("Position Vs. Intensity fit for horizontal slice")
            plt.show()
           
            plt.plot(np.linspace(-0.05, 0.05,1000), squarefit(np.linspace(-0.05,0.05,1000), *popt2s))
            plt.plot(sintheta, middleydat, "+")
            plt.title("Position Vs. Intensity for vertical slice")
            plt.show() """
           
    elif min(chi_for_diag,chi_for_square,chi_for_circ) == chi_for_diag:
           shape = "diamond"
           poptfin1, pcovfin1 = popt1d, pcov1d
           poptfin2, pcovfin2 = popt2d, pcov2d
           
           """#plot diamond
           plt.plot(np.linspace(-0.05, 0.05, 1000), diafit(np.linspace(-0.05,0.05,1000), *popt1d))
           plt.plot(sinphi, middlexdat, "+")
           plt.title("Position Vs. Intensity fit for diamond")
           plt.show()
           
           plt.plot(np.linspace(-0.05, 0.05, 1000), diafit(np.linspace(-0.05,0.05,1000), *popt2d))
           plt.plot(sintheta, middleydat, "+")
           plt.title("Position Vs. Intensity fit for diamond 2")
           plt.show()"""


   
   
    dimensions = abs(10*poptfin1[1]), abs(10*poptfin2[1])
    if shape == "circle":
        Area = np.pi * (0.5 * np.average(dimensions))**2
    else:
        Area = dimensions[0] * dimensions[1]
   
    #convert the pcovs into standard deviations
    perr1 = np.sqrt(np.diag(pcovfin1))
    perr2 = np.sqrt(np.diag(pcovfin2))
   
    #I0 is the average of the verticle and horizontal values for I0
    I0 = 0.5*(poptfin1[0] + poptfin2[0])
   
    #calculating the error in the area
    area_err = squ_error(dimensions[0], dimensions[1], perr1[1], perr2[1])
    #calculating error in the intensity per unit area (relative to the intensity per unit area)
    intensity_per_area_err = squ_error(1, I0, area_err, 0.5*(perr1[0]+perr2[0]))
    #calculating the intensity per unit area
    intensity_per_unit_area = I0 / Area
    #converting it back to the same units as intesnsity per unit area
    IperA_fin = intensity_per_unit_area*intensity_per_area_err
   
   
    # plotting everything on one graph
    plt.plot(sinphi, middlexdat, "+", label = "data")
    plt.plot(np.linspace(-0.05, 0.05, 1000), squarefit(np.linspace(-0.05,0.05,1000), *popt1d),label = "square")
    plt.plot(np.linspace(-0.05, 0.05, 1000), diafit(np.linspace(-0.05,0.05,1000), *popt1d), label = "diamond")
    plt.plot(np.linspace(-0.05, 0.05, 1000),circfit(np.linspace(-0.05,0.05,1000), *popt1c), label = "circle*")
    plt.title(r"Slice through $\theta = 0$")
    plt.legend()
    plt.xlabel("sin angle")
    plt.ylabel("Intensity")
    plt.annotate("$I_0 = 239 \pm 0.9 n$" ,xy = (0.2,0.7), xycoords = "figure fraction")
    plt.annotate("$D = 19.7 \pm 0.007 \mu m$", xy = (0.2, 0.65), xycoords = "figure fraction")
    plt.show()
   
    plt.plot(sintheta, middleydat, "+", label = "data")
    plt.plot(np.linspace(-0.05, 0.05, 1000), squarefit(np.linspace(-0.05,0.05,1000), *popt2d),label = "square")
    plt.plot(np.linspace(-0.05, 0.05, 1000), diafit(np.linspace(-0.05,0.05,1000), *popt2d), label = "diamond")
    plt.plot(np.linspace(-0.05, 0.05, 1000),circfit(np.linspace(-0.05,0.05,1000), *popt2c), label = "circle")
    plt.title(r"Slice through $\phi = 0$")
    plt.legend()
    plt.xlabel("sin angle")
    plt.ylabel("Intensity")
    plt.annotate("$I_0 = 239 \pm 0.9 n$" ,xy = (0.2,0.7), xycoords = "figure fraction")
    plt.annotate("$D = 19.5 \pm 0.009 \mu m$", xy = (0.2, 0.65), xycoords = "figure fraction")
    plt.show()
   

    results = {
        "shape": shape, # one of "square", "rectangle", "diamond", "circle" - must always be present.
        "dim_1": dimensions[0] *10**6, # a floating point number of the first dimension expressed in microns
        "dim_1_err": perr1[1] *10**6, # The uncertainty in the above, also expressed in microns
        "dim_2": dimensions[1] *10**6, # For a rectangle, the second dimension, for other shapes, the same as dim_1
        "dim_2_err": perr2[1] *10**6,  # The uncertainty in the above, also expressed in microns
        "I0/area": intensity_per_unit_area, # The fitted overall intensity value/area of the aperture.
        "I0/area_err": IperA_fin,# The uncertainty in the above.
    }
    return results
   
results = ProcessData("C:/Users/py22ec/Downloads/assessment_data_py22ec.dat")
print(results)
