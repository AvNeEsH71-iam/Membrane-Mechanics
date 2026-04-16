import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numba import jit
import imageio
import os

# data extraction
def extract(filename):
    data = []
    with open(f"build/{filename}", "r") as f:
        data = f.readlines()
    f.close()

    data.pop(0)
    data_new = []
    for i in range(len(data)):
        if len(data[i].split(",")[:-1]) == len(data[1].split(",")[:-1]):
            data_new.append(data[i].split(",")[:-1])

    return np.asarray(data_new).astype("float64")


with open("build/mean_radius.txt", "r") as f:
    mean_radius = f.readlines()
f.close()
mean_radius = np.asarray(mean_radius).astype("float64")
plt.plot(mean_radius)
plt.title(r"$r_{avg}\ vs.\ time(t)$", fontsize=15)
plt.ylabel(r"$r_{avg}$", fontsize=12)
plt.xlabel(r"$time (t)$", fontsize=12)
plt.tight_layout()
plt.savefig("plots/mean_radius.png", dpi=300)
plt.show()


# angular autocorrelation data
epsilon = extract("epsilon.txt")
gamma = extract("gamma.txt")

# angular correlation for specific frame number
frame_number = int(
    input(
        "Enter the frame number for which you would like to see Angular Autocorrelation graph: "
    )
)
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["figure.figsize"] = (10, 6)
plt.plot(gamma[frame_number], epsilon[frame_number])
plt.title(rf"$\epsilon$ vs. $\gamma$ (frame={frame_number})", fontsize=20)
plt.xlabel(r"$\gamma$ (in radians)", fontsize=15)
plt.ylabel(r"$\epsilon$", fontsize=15)
plt.tight_layout()
plt.savefig("plots/angular_autocorrelation.png", dpi=300)
plt.show()

# creating gifs (takes 1m 30s for 1000 frames) !!!
def create_gif(start, stop):
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams["figure.figsize"] = (10, 6)

    filenames = []

    for i in range(start, stop):
        plt.plot(gamma[i], epsilon[i])
        plt.title(rf"$\epsilon$ vs. $\gamma$ (frame = {i})", fontsize=20)
        plt.xlabel(r"$\gamma$ (in radians)", fontsize=15)
        plt.ylabel(r"$\epsilon$", fontsize=15)
        plt.tight_layout()

        # create file name and append it to a list
        filename = f"{i}.png"
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()

    # build gif
    with imageio.get_writer("angular_correlation.gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


gif = int(
    input(
        "\nIf you want to create a gif of Angular Autocorrelation enter '1' else '0'\nIt takes around 1m 30s for 1000 frames: "
    )
)
if gif == 1:
    start = int(input("\nEnter first frame number for gif: "))
    stop = int(input("Enter final frame number for gif: "))
    create_gif(start, stop)


fourier_start = int(input("\nEnter first mode to be analyzed for Fourier: "))
fourier_stop = int(input("\nEnter last mode to be analyzed for Fourier: ")) + 1

legendre_start = 2
legendre_stop = 11

# hard thresholding
def hard_threshold(data, type):

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    thres_dict = {}

    for k in range(start, stop):

        threshold = 1
        gpu_data = cp.asarray(data[:, k])

        gpu_data = cp.where(
            cp.logical_or(gpu_data < -threshold, gpu_data > +threshold),
            0 * gpu_data,
            gpu_data,
        )
        mean = cp.mean(gpu_data)
        new_data = cp.asnumpy(gpu_data)
        data[:, k] = new_data
        label_mean = round(cp.asnumpy(mean).item(), 7)

        fig, ax = plt.subplots()

        if type == "fourier":
            plt.plot(new_data, label=rf"$k = {k}$")  # todo: scaling
            plt.plot(
                range(len(new_data)),
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="red",
                label=rf"$<a> = {label_mean}$",
            )
            plt.ylabel(r"$a_k$")
            plt.title(f"{type} amplitude vs. time")
        else:
            plt.plot(new_data, label=rf"$l = {k}$")  # scaling
            plt.plot(
                range(len(new_data)),
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="red",
                label=rf"$<b> = {label_mean}$",
            )
            plt.ylabel(r"$b_l$")
            plt.title(f"{type} amplitude vs. time")

        plt.xlabel("time (t)")

        (ut,) = plt.plot(
            range(len(new_data)),
            cp.asnumpy(cp.zeros(len(new_data)) + (mean + cp.std(gpu_data))),
            color="black",
        )
        (lt,) = plt.plot(
            range(len(new_data)),
            cp.asnumpy(cp.zeros(len(new_data)) + (mean - cp.std(gpu_data))),
            color="black",
        )

        plt.subplots_adjust(bottom=0.40)
        thres_dict[k] = [0, 0]

        axthresup = plt.axes([0.2, 0.15, 0.65, 0.03])
        sthresup = Slider(
            axthresup,
            "Upper Threshold",
            cp.asnumpy(mean),
            np.max(new_data),
            valinit=cp.asnumpy(mean + cp.std(gpu_data)),
            valstep=(np.max(new_data) - cp.asnumpy(mean)) / 10000,
        )

        def update_up(val):
            ut.set_ydata(cp.asnumpy(cp.zeros(len(new_data)) + sthresup.val))
            thres_dict[k][1] = sthresup.val
            fig.canvas.draw_idle()

        sthresup.on_changed(update_up)

        axthresbottom = plt.axes([0.2, 0.05, 0.65, 0.03])
        if (np.min(new_data)) < 0.0:
            sthresbottom = Slider(
                axthresbottom,
                "Lower Threshold",
                np.min(new_data),
                cp.asnumpy(mean),
                valinit=cp.asnumpy(mean - cp.std(gpu_data)),
                valstep=(cp.asnumpy(mean) + np.min(new_data)) / 10000,
            )

            def update_bottom(val):
                lt.set_ydata(cp.asnumpy(cp.zeros(len(new_data)) + sthresbottom.val))
                thres_dict[k][0] = sthresbottom.val
                fig.canvas.draw_idle()

        else:
            sthresbottom = Slider(
                axthresbottom,
                "Lower Threshold",
                np.min(new_data),
                cp.asnumpy(mean),
                valinit=cp.asnumpy(mean - cp.std(gpu_data)),
                valstep=(cp.asnumpy(mean) - np.min(new_data)) / 10000,
            )

            def update_bottom(val):
                lt.set_ydata(cp.asnumpy(cp.zeros(len(new_data)) + sthresbottom.val))
                thres_dict[k][0] = sthresbottom.val
                fig.canvas.draw_idle()

        sthresbottom.on_changed(update_bottom)

        ax.legend(loc="upper left")
        plt.savefig(f"plots/{type}_amplitude_{k}.png", dpi=300)
        plt.show()

    return data, thres_dict


# soft thresholding
def soft_threshold(data, type, thres_dict):

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    for l in range(start, stop):
        gpu_data = cp.asarray(data[:, l])
        upper_thres = thres_dict[l][1]
        lower_thres = thres_dict[l][0]
        gpu_data = cp.where(
            cp.logical_or(gpu_data < lower_thres, gpu_data > upper_thres),
            0 * gpu_data,
            gpu_data,
        )
        new_data = cp.asnumpy(gpu_data)
        data[:, l] = new_data
        mean = cp.mean(gpu_data)
        label_mean = round(cp.asnumpy(mean).item(), 7)
        if type == "fourier":
            plt.plot(new_data, label=rf"$k = {l}$")
            plt.plot(
                range(len(new_data)),
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="red",
                label=rf"$<a> = {label_mean}$",
            )
            plt.ylabel(r"$a_k$")
            plt.title(f"{type} amplitude vs. time")
        else:
            plt.plot(new_data, label=rf"$l = {l}$")
            plt.plot(
                range(len(new_data)),
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="red",
                label=rf"$<b> = {label_mean}$",
            )
            plt.ylabel(r"$b_l$")
            plt.title(f"{type} amplitude vs. time")
        plt.xlabel("time (t)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots/{type}_threshold_{l}.png", dpi=300)
        plt.show()

    return data


# chi square function
@jit
def chi_square(kappa, sigma, means, deviations, type):
    sum = 0
    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    for l in range(start, stop):
        b_t = (2 * l + 1) / (
            4 * np.pi * kappa * (l - 1) * (l + 2) * (l * (l + 1) + sigma)
        )
        sum += ((means[l - start] - b_t) / (deviations[l - start])) ** 2
    return sum


# monte carlo
def monte_carlo(data, type):
    steps = 1000000

    states = []
    chi_history = []
    kappa = 40
    sigma = 5
    states.append((kappa, sigma))

    means = []
    deviations = []

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    for l in range(start, stop):
        means.append(np.mean(data[l]))
        deviations.append(np.std(data[l]))

    means = np.asarray(means)
    deviations = np.asarray(deviations)

    for i in range(steps):
        old_chi = chi_square(kappa, sigma, means, deviations, type)
        chi_history.append(old_chi)

        old_kappa = kappa
        old_sigma = sigma

        delta_kappa = np.random.uniform(-0.1, 0.1)
        delta_sigma = np.random.uniform(-0.05, 0.05)

        kappa += delta_kappa
        sigma += delta_sigma

        while kappa < 0 or kappa > 50:
            if kappa < 0:
                delta_kappa = np.random.uniform(0, 0.1)
                kappa += delta_kappa
            if kappa > 50:
                delta_kappa = np.random.uniform(-0.1, 0)
                kappa += delta_kappa

        while sigma < -10 or sigma > 10:
            if sigma < -10:
                delta_sigma = np.random.uniform(0, 0.1)
                sigma += delta_sigma
            if sigma > 10:
                delta_sigma = np.random.uniform(-0.1, 0)
                sigma += delta_sigma

        delta_chi = chi_square(kappa, sigma, means, deviations, type) - old_chi

        if delta_chi <= 0:
            # accept
            states.append((kappa, sigma))
        else:
            if np.exp(-delta_chi) >= np.random.uniform(0.0, 1.0):
                # accept
                states.append((kappa, sigma))
            else:
                # reject
                kappa = old_kappa
                sigma = old_sigma

    return (states, means, deviations, chi_history)


# Fourier Analysis
plt.rcParams["figure.figsize"] = (10, 3)
# fourier amplitude data
a = extract("fourier_amplitudes.txt")
a, thres_dict = hard_threshold(a, "fourier")

a = np.copy(soft_threshold(a, "fourier", thres_dict))
a = a.T
print("\nProcessing...", end="\r")
states, means, deviations, chi_history = monte_carlo(a, "fourier")
kappa = states[-1][0]
sigma = states[-1][1]
print("Fourier Analysis Results:\n")
print("kappa =", kappa)
print("sigma =", sigma)

plt.plot(chi_history)
plt.title(r"$\chi^2$ vs. iterations")
plt.ylabel(r"$\chi^2$")
plt.xlabel("iterations")
plt.savefig("plots/chi_fourier.png", dpi=300)
plt.show()

states = np.asarray(states)
plt.plot(states[:, 0])
plt.title("Bending Rigidity")
plt.ylabel(r"$\kappa$")
plt.xlabel("iterations")
plt.savefig("plots/kappa_fourier.png", dpi=300)
plt.show()

# theoretical vs. experimental fourier amplitude
a_t = []
for l in range(fourier_start, fourier_stop):
    a_t.append(
        (2 * l + 1) / (4 * np.pi * kappa * (l - 1) * (l + 2) * (l * (l + 1) + sigma))
    )
plt.plot(range(fourier_start, fourier_stop), a_t, color="red", label="theoretical")
plt.errorbar(
    range(fourier_start, fourier_stop),
    means,
    deviations,
    ecolor="black",
    color="blue",
    label="experimental",
)
plt.title("Fourier Amplitude verification")
plt.ylabel(r"$a_k$")
plt.xlabel(r"$k$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/fourier_verification.png", dpi=300)
plt.show()

# Legendre Analysis

# legendre amplitude data
b = extract("legendre_amplitudes.txt")
b, thres_dict = hard_threshold(b, "legendre")

b = np.copy(soft_threshold(b, "legendre", thres_dict))
b = b.T
print("\nProcessing...", end="\r")
states, means, deviations, chi_history = monte_carlo(b, "legendre")
kappa = states[-1][0]
sigma = states[-1][1]
print("Legendre Analysis Results:\n")
print("kappa =", kappa)
print("sigma =", sigma)

plt.plot(chi_history)
plt.title(r"$\chi^2$ vs. iterations")
plt.ylabel(r"$\chi^2$")
plt.xlabel("iterations")
plt.savefig("plots/chi_legendre.png", dpi=300)
plt.show()

states = np.asarray(states)
plt.plot(states[:, 0])
plt.title("Bending Rigidity")
plt.ylabel(r"$\kappa$")
plt.xlabel("iterations")
plt.savefig("plots/kappa_legendre.png", dpi=300)
plt.show()

# theoretical vs. experimental legendre amplitude
b_t = []
for l in range(legendre_start, legendre_stop):
    b_t.append(
        (2 * l + 1) / (4 * np.pi * kappa * (l - 1) * (l + 2) * (l * (l + 1) + sigma))
    )
plt.plot(range(legendre_start, legendre_stop), b_t, color="red", label="theoretical")
plt.errorbar(
    range(legendre_start, legendre_stop),
    means,
    deviations,
    ecolor="black",
    color="blue",
    label="experimental",
)
plt.title("Legendre Amplitude verification")
plt.ylabel(r"$b_l$")
plt.xlabel(r"$l$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/legendre_verification.png", dpi=300)
plt.show()


# time autocorrelation
time = []
with open("build/time_auto_correlation.txt", "r") as f:
    time = f.readlines()
f.close()

for i in range(len(time)):
    time[i] = time[i].split(",")[:-1]

time = np.asarray(time).astype("float64")

for l in range(legendre_start, legendre_stop):
    plt.plot(time[l, :60], label=rf"$l={l}$")
plt.legend(bbox_to_anchor=(1.15, 1.05))
plt.title("Time Autocorrelation")
plt.ylabel("C(t)")
plt.xlabel("time (t)")
plt.tight_layout()
plt.savefig("plots/time_autocorrelation.png")
plt.show()

for l in range(legendre_start, legendre_stop):
    plt.plot(np.log(time[l, :60]), label=rf"$l={l}$")
plt.legend(bbox_to_anchor=(1.15, 1.05))
plt.title("Time Autocorrelation")
plt.ylabel("log(C(t))")
plt.xlabel("time (t)")
plt.tight_layout()
plt.savefig("plots/time_autocorrelation_log.png")
plt.show()

# time cross correlation
time = []
with open("build/time_cross_correlation.txt", "r") as f:
    time = f.readlines()
f.close()

for i in range(len(time)):
    time[i] = time[i].split(",")[:-1]

time = np.asarray(time).astype("float64")

modes = []
with open("build/time_cross_input.txt", "r") as f:
    modes = f.readlines()
f.close()

for i in range(len(modes)):
    modes[i] = modes[i].strip().split(",")

for l in range(len(time)):
    plt.plot(time[l, :60], label=rf"$l={int(modes[l][0]),int(modes[l][1])}$")
plt.legend(bbox_to_anchor=(1.18, 1.05))
plt.title("Time Crosscorrelation")
plt.ylabel("C(t)")
plt.xlabel("time (t)")
plt.tight_layout()
plt.savefig("plots/time_crosscorrelation.png")
plt.show()

for l in range(len(time)):
    plt.plot(np.log(time[l, :60]), label=rf"$l={int(modes[l][0]),int(modes[l][1])}$")
plt.legend(bbox_to_anchor=(1.18, 1.05))
plt.title("Time Crosscorrelation")
plt.ylabel("log(C(t))")
plt.xlabel("time (t)")
plt.tight_layout()
plt.savefig("plots/time_crosscorrelation_log.png")
plt.show()
