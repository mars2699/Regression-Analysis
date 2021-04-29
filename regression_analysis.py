"""
Created on Mon Mar  8 22:01:59 2021

@author: mars2699
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

# Define and print the data

years = np.array([1970,1973,1976,1979,1982,1985,1988,1991,1994,1997,2000,2003,2007,2010,2013,2016,2019], dtype=np.float64)
passengers = np.array([163448992,202309200,223017296,313624000,290992608,372059104,454202912,452015904,514924000,590571392,665327414,588997110,744302310,720497000,743171000,824039000,926737000], dtype=np.float64)
# dtype describes how the array memory should be interpreted. Float64 uses 64 bits.

print('Years: \n', years)
print('Passengers: \n', passengers)

# Find the value of r, the correlation coefficient

r = np.corrcoef(years, passengers)     # This calculates the correlation matrix, like you can do in Excel
rCorrel = r[0,1]                   # Arranges the value into a 0 by 1 matrix

# From here we can find R^2, the coefficient of determination
rSquared = rCorrel**2               
print('\nHere is the correlation matrix for R^2: \n', rSquared)

# When I was originally doing this assignment, I was aiming for an R^2 value
# of at least 0.7. Let's tell the user if their data is good to continue 
# with by this standard

if rSquared >=0.7:
    print('\nThis data seems to have a fairly high correlation. You are good to go!') 
else:
    print('\nHmm...this data seems to have a low correlation. Maybe try another data set.')

# Plot data
# I was having trouble getting the axis labels to display without being in
# scientific notation. On stackoverflow: https://stackoverflow.com/questions/28371674/prevent-scientific-notation-in-matplotlib-pyplot
# There is an example similar to my problem that I used to fix this.

fig, ax = plt.subplots()   # Fig and ax store figure and axis objects. We "seperate" our axis from the rest of the graph
ax.plot(years, passengers, 'r*')
ax.plot(years, passengers, 'b-')
ax.ticklabel_format(useOffset=False, style='plain')  # We don't want to use scientific notation
plt.title('Air Transport, Passengers Carried (USA)')
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.grid(True)

plt.show()

# Now I added in some predictor variables. I took into account the average gas price per gallon,
# the average yearly household income, the gdp per capita, and the average roundtrip domestic
# airline ticket price.

xGasArray = np.array([1.72, 1.67, 1.59, 1.62, 2.03, 1.98, 1.96, 1.94, 1.83, 2.31, 2.95, 2.97, 2.6, 2.37, 2.23, 2.14, 1.61, 1.64, 1.59, 1.7, 1.89, 1.81, 1.75, 1.68, 1.65, 1.67, 1.76, 1.74, 1.47, 1.6, 2.02, 1.91, 1.75, 2.01, 2.32, 2.74, 3, 3.16, 3.61, 2.58, 3.02, 3.75, 3.8, 3.62, 3.4, 2.45, 2.14, 2.42, 2.72, 2.6])

xIncomeArray = np.array([65224.83, 65849.49, 44147.98, 45885.13, 56251.55, 65018.96, 67606.35, 57473.43, 58507.99, 56683.46, 64062.04, 62647.70, 63139.53, 63819.19, 66015.76, 56837.98, 56363.54, 59235.13, 58825.65, 62689.52, 53788.13, 51426.01, 49474.06, 48621.51, 47936.37, 48033.04, 47473.53, 48728.52, 50335.33, 49514.77, 48418.76, 42228.00, 42409.00, 43318.00, 44031.00, 46242.00, 46326.00, 50233.00, 52029.00, 57010.00, 49445.00, 50054.00, 54569.00, 52250.00, 53657.00, 55775.00, 57617.00, 60336.00, 63179.00, 68703.00])

xGDPArray = np.array([5234.297, 5609.383, 6094.018, 6726.359, 7225.691, 7801.457, 8592.254, 9452.577, 10564.948, 11674.186, 12574.792, 13976.11, 14433.788, 15543.894, 17121.225, 18236.828, 19071.227, 20038.941, 21417.012, 22857.154, 23888.6, 24342.259, 25418.991, 26387.294, 27694.853, 28690.876, 29967.713, 31459.139,	32853.677, 34513.562, 36334.909, 37133.243,	38023.161, 39496.486, 41712.801, 44114.748,	46298.731, 47975.968, 48382.558, 47099.98, 48467.516, 49886.818, 51610.605,	53117.668, 55064.745, 56839.382, 57951.584,	60062.222, 62996.471, 65297.518])

yPassengersArray = np.array([163448992, 174143104, 191325408, 202309200, 207612400, 204900400, 223017296, 240144992, 273025504, 313624000,	295329088, 281086400, 290992608, 315600096, 340191488, 372059104, 414554496, 441832704, 454202912, 453161504, 464574016, 452015904, 466964992, 469926112, 514924000, 533512096, 571072000, 590571392, 588170880, 634364608, 665327414, 622187846, 598410415, 588997110, 678110608, 720547738, 725530965, 744302310, 701779551, 679423408, 720497000, 730796000,	736699000, 743171000, 762710000, 798222000, 824039000, 849403000, 889024000, 926737000])

xTicketArray = np.array([550.00, 558.00, 570.00, 600.00, 610.00, 628.00, 647.00, 671.00, 680.00, 615.00, 620.00, 600.00, 550.00, 555.00, 560.00, 500.00, 460.00, 455.00, 470.00, 480.00, 500.00, 470.00, 450.00, 460.00, 410.00, 407.00, 405.00, 403.00, 410.00, 400.00, 420.00, 360.00, 348.00, 347.00, 315.00, 310.00, 345.00, 330.00, 347.00, 300.00, 340.00, 360.00, 381.00, 383.00, 392.00, 399.00, 344.00, 417.00, 346.00, 359.00])

# I normalized the data
normalized_arr1 = preprocessing.normalize([xGasArray])
normalized_arr2 = preprocessing.normalize([xIncomeArray])
normalized_arr3 = preprocessing.normalize([xGDPArray])
normalized_arr4 = preprocessing.normalize([yPassengersArray])
normalized_arr5 = preprocessing.normalize([xTicketArray])

# To visually see the correlations of each predictor, I made a color-coded
# scatter plot of each normalized predictor

plt.scatter(normalized_arr3, normalized_arr4, c='g', label = 'GDP per Capita')
plt.scatter(normalized_arr5, normalized_arr4, c='m', label = 'Avg Ticket Price')
plt.scatter(normalized_arr2, normalized_arr4, c='r', label = 'Avg Income')
plt.scatter(normalized_arr1, normalized_arr4, c='b', label = 'Avg Gas Price')

plt.legend(loc = 'lower center', numpoints = 1, ncol = 3, fontsize = 10, bbox_to_anchor = (0.5, -0.3))
plt.title('Predictor Correlations Visualized')
plt.show()

# Find the R^2 of each predictor and show it to the user

rOfGas = np.corrcoef(normalized_arr4, normalized_arr1)
rOfGasCorrel = rOfGas[0,1]
rOfGasSquared = rOfGasCorrel**2
print('\nR^2 of Average Gas Price per Gallon: ', rOfGasSquared)

rOfIncome = np.corrcoef(normalized_arr4, normalized_arr2)
rOfIncomeCorrel = rOfIncome[0,1]
rOfIncomeSquared = rOfIncomeCorrel**2
print('\nR^2 of Average Yearly Household Income: ', rOfIncomeSquared)

rOfGDP = np.corrcoef(normalized_arr4, normalized_arr3)
rOfGDPCorrel = rOfGDP[0,1]
rOfGDPSquared = rOfGDPCorrel**2
print('\nR^2 of GDP per capita: ', rOfGDPSquared)

rOfTicket = np.corrcoef(normalized_arr4, normalized_arr5)
rOfTicketCorrel = rOfTicket[0,1]
rOfTicketSquared = rOfTicketCorrel**2
print('\nR^2 of Average Roundtrip Domestic Airline Ticket Price: ', rOfTicketSquared)

sys.exit()
