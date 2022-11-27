import pandas as pd
import matplotlib.pyplot as plt

def bins_plot(processed_data_path):

    signal = pd.read_csv(processed_data_path)

    bins = ((.1, .98), (1.1, 2.5), (2.5, 4.), (4., 6.), (6., 8.), (15., 17.), \
            (17., 19.), (11., 12.5), (1., 6.), (15., 17.9))
    freqs = [0 for i in range(0, len(bins))]
    bin_nums = list(range(0, len(bins)))

    q2 = signal["q2"]
    # print(q2.head())
    for i in range(len(q2)):
        for j in range(len(bins)):
            bin = bins[j]
            q2_i = q2.iloc[i]
            if bin[0] < q2_i and q2_i < bin[1]:
                freqs[j] = freqs[j] + 1
            # print(q2_i, freqs)

    plt.grid()
    plt.bar(bin_nums, freqs)
    plt.ylabel("Frequency")
    plt.xlabel("Bin")
    plt.show()

if __name__ == "__main__":

    path = "D:/Projekty/Coding/Python/TBPSWeAreNumberOne/data/acceptance_mc.csv"

    bins_plot(path)