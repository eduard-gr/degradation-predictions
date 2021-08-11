import matplotlib.pyplot as plt

# create figure (will only create new window if needed)
plt.figure()
# Generate plot1
plt.title("title1")
plt.plot(range(10, 20),label="label1")

# Show the plot in non-blocking mode
plt.show(block=False)

# create figure (will only create new window if needed)
plt.figure()
# Generate plot2
plt.title("title2")
plt.plot(range(10, 20),label="label1")

# Show the plot in non-blocking mode
plt.show(block=False)

...

# Finally block main thread until all plots are closed
plt.legend()
plt.show()