import sys
import numpy as np

def load_data():
	try:
		data = np.genfromtxt(sys.argv[1], delimiter = ',')
	except:
		print("Error")
		exit()
	return data

def sum_up_stats(tmp_stats, sanitized_col):
	tmp_stats[0][1] /= tmp_stats[0][0]
	tmp_stats[0][2] = (sum((sanitized_col - tmp_stats[0][1]) ** 2) / \
		tmp_stats[0][0]) ** 0.5
	for i in range(1, 4):
		percentile = (tmp_stats[0][0] - 1) * i * 0.25
		decimal = percentile - np.floor(percentile)
		percentile = int(np.floor(percentile))
		if tmp_stats[0][0] == 1:
			tmp_stats[0][i + 3] = sanitized_col[percentile]
		else:
			tmp_stats[0][i + 3] = sanitized_col[percentile] + \
				(sanitized_col[percentile + 1] - sanitized_col[percentile]) * decimal
	return tmp_stats

def display_stats(stats):
	labels = ["", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
	lines, columns = np.shape(stats)
	for i in range(0, lines + 1):
		sys.stdout.write("%-8s" % labels[i])
		for j in range(0, columns):
			if i == 0:
				sys.stdout.write("%12s %2d " % ("Feature", j + 1))
			else:
				sys.stdout.write("%15.6f " % stats[i - 1][j])
		sys.stdout.write("\n")


def main():
	data = load_data()
	try:
		lines, columns = np.shape(data)
	except:
		print("Error")
		exit()
	for i in range(0, columns):
		tmp_stats = np.zeros((1, 8))
		tmp_stats[0][3] = np.inf
		tmp_stats[0][7] = -np.inf
		for j in range(0, lines):
			if ~np.isnan(data[j][i]):
				tmp_stats[0][0] += 1
				tmp_stats[0][1] += data[j][i]
				if data[j][i] < tmp_stats[0][3]:
					tmp_stats[0][3] = data[j][i]
				if data[j][i] > tmp_stats[0][7]:
					tmp_stats[0][7] = data[j][i]
		if tmp_stats[0][0]:
			sanitized_col = np.sort(np.extract(~np.isnan(data[:, i]), data[:, i]))
			tmp_stats = sum_up_stats(tmp_stats, sanitized_col)
			if i:
				stats = np.vstack([stats, tmp_stats])
			else:
				stats = tmp_stats
	display_stats(stats.T)

if __name__ == "__main__":
	main()
