import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tk

def get_col(data, data_str, column):
	ravenclaw = np.array([])
	slytherin = np.array([])
	gryffindor = np.array([])
	hufflepuff = np.array([])
	for k in range(1, len(data)):
		if data_str[k, 1] == "Ravenclaw":
			ravenclaw = np.append(ravenclaw, data[k, column])
		elif data_str[k, 1] == "Slytherin":
			slytherin = np.append(slytherin, data[k, column])
		elif data_str[k, 1] == "Gryffindor":
			gryffindor = np.append(gryffindor, data[k, column])
		elif data_str[k, 1] == "Hufflepuff":
			hufflepuff = np.append(hufflepuff, data[k, column])
	return ravenclaw, slytherin, gryffindor, hufflepuff

def get_data():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print("Error")
		exit()
	best_hand = np.where(data_str == "Best Hand")[1]
	data_str[np.where(data_str == "Left")[0], best_hand] = 1
	data_str[np.where(data_str == "Right")[0], best_hand] = 2
	data[1:, best_hand] = data_str[1:, best_hand]
	birthday = np.where(data_str == "Birthday")[1]
	for i in range(1, len(data_str)):
		nb_bday = data_str[i, birthday]
		data[i, birthday] = int(nb_bday[0][:4]) + \
			(int(nb_bday[0][5:7]) / 120 * 10) + \
			(int(nb_bday[0][8:9]) / 3100 * 10)
	return data, data_str

def set_ticks(data, data_str, axarr, i, j, houses_x, houses_y):
	concat = np.concatenate(houses_x)
	axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].set_xlim([min(concat) - 1, \
		max(concat) + 1])
	if i != j:
		concat = np.concatenate(houses_y)
		axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].set_ylim(\
			[min(concat) - 1, max(concat) + 1])
	if i != data.shape[1] - 1:
		axarr[i - 4, j - 4].xaxis.set_minor_locator(tk.NullLocator())
		axarr[i - 4, j - 4].xaxis.set_major_locator(tk.NullLocator())
	else:
		axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].tick_params(labelsize = \
			"xx-small")
		axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].set_xlabel(\
			data_str[0, j], fontsize = 6.5)
	if j != 4:
		axarr[i - 4, j - 4].yaxis.set_minor_locator(tk.NullLocator())
		axarr[i - 4, j - 4].yaxis.set_major_locator(tk.NullLocator())
	else:
		if i == data.shape[1] - 1:
			axarr[i - 4, j - 4].yaxis.set_minor_locator(tk.NullLocator())
			axarr[i - 4, j - 4].yaxis.set_major_locator(tk.NullLocator())
		else:
			axarr[i - 4, 0].tick_params(labelsize = "xx-small")
		axarr[i - 4, 0].set_ylabel(data_str[0, i], fontsize = 5.5)

def main():
	data, data_str = get_data()
	fig, axarr = plt.subplots(data.shape[1] - 4, data.shape[1] - 4)
	plt.subplots_adjust(wspace = 0, hspace = 0, left = 0.07, right = 0.92, \
		top = 0.95, bottom = 0.05)
	colors = ['r', 'b', 'g', 'y']
	for i in range(4, data.shape[1]):
		for j in range(data.shape[1] - 1, 3, -1):
			houses_x = get_col(data, data_str, j)
			houses_y = get_col(data, data_str, i)
			if i != j:
				for k in range(0, 4):
					axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].scatter(\
						houses_x[k], houses_y[k], c = colors[k], \
						edgecolors = "none", s = 2)
			else:
				concat = np.concatenate(houses_x)
				axarr[i - 4, (j - 4) * -1 + data.shape[1] - 5].hist( \
					np.extract(~np.isnan(concat), concat), bins = 40)
			set_ticks(data, data_str, axarr, i , j, houses_x, houses_y)
	plt.legend(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"], \
		fontsize = "x-small", loc = (1.04, 0))
	plt.show()

if __name__ == "__main__":
	main()
