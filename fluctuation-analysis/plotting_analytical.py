import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numba import jit
import imageio
import os

fps = 24
#fps is the scaling factor for frames to seconds conversion for camera settings for recording.
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
plt.plot(np.arange(0, len(mean_radius)) / fps, mean_radius * 0.23, color="black")
#0.23 is the scaling factor for pixels to micron conversion for 20X, no binning.
# plt.title(r"$r_{avg}\ vs.\ time(t)$", fontsize=15)
plt.ylabel(r"$r_{avg}\ (\mu m)$", fontsize=15)
plt.xlabel(r"$time (s)$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
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
plt.plot(gamma[frame_number], epsilon[frame_number], color="black")
# plt.title(rf"$\epsilon$ vs. $\gamma$ (frame={frame_number})", fontsize=15)
plt.xlabel(r"$\gamma$ (in radians)", fontsize=15)
plt.ylabel(r"$\xi$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
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
        plt.title(rf"$\epsilon$ vs. $\gamma$ (frame = {i})", fontsize=15)
        plt.xlabel(r"$\gamma$ (in radians)", fontsize=15)
        plt.ylabel(r"$\epsilon$", fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
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

legendre_start = int(input("\nEnter first mode to be analyzed for Legendre: "))
legendre_stop = int(input("\nEnter last mode to be analyzed for Legendre: ")) + 1

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
            # plt.title(f"{type} amplitude vs. time")
        else:
            plt.plot(new_data, label=rf"$l = {k}$")  # scaling
            plt.plot(
                range(len(new_data)),
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="red",
                label=rf"$<b> = {label_mean}$",
            )
            plt.ylabel(r"$b_l$")
            # plt.title(f"{type} amplitude vs. time")

        plt.xlabel("time (t)")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

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
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.plot(
                np.arange(0, len(new_data)) / fps,
                new_data,
                label=rf"$k = {l}$",
                color="white",
            )
            plt.plot(
                np.arange(0, len(new_data)) / fps,
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="black",
                label=rf"$<a> = {label_mean}$",
            )
            plt.ylabel(r"$a_k$", fontsize=15)
            # plt.title(f"{type} amplitude vs. time")
        else:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.plot(
                np.arange(0, len(new_data)) / fps,
                new_data,
                label=rf"$l = {l}$",
                color="white",
            )
            plt.plot(
                np.arange(0, len(new_data)) / fps,
                cp.asnumpy(cp.zeros(len(new_data)) + (mean)),
                color="black",
                label=rf"$<b> = {label_mean}$",
            )
            plt.ylabel(r"$b_l$", fontsize=15)
            # plt.title(f"{type} amplitude vs. time")
        plt.xlabel(r"time (s)", fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots/{type}_threshold_{l}.png", dpi=300)
        plt.show()

    return data


def q(l):
    return 4 * np.pi * (l - 1) * (l + 2) / (2 * l + 1)


def p(l):
    return 4 * np.pi * (l - 1) * (l + 2) * (l + 1) * l / (2 * l + 1)


def r(s, means, deviations, type):

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    num = 0
    den = 0

    for l in range(start, stop):
        num += (p(l) * means[l - start]) / (
            ((p(l) * deviations[l - start]) ** 2) * (1 + s * q(l) / p(l))
        )
        den += 1 / (((p(l) * deviations[l - start]) * (1 + s * q(l) / p(l))) ** 2)

    return num / den


# chi square function
def chi_square_func(s, means, deviations, type):

    chi_square = 0

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    for l in range(start, stop):
        chi_square += (
            (
                p(l) * means[l - start]
                - (r(s, means, deviations, type) / (1 + s * q(l) / p(l)))
            )
            / (p(l) * deviations[l - start])
        ) ** 2

    return chi_square


# gradient descent
# def gradient_descent(data, type):

#     steps = 1000
#     chi_history = []

#     s = 5
#     means = []
#     deviations = []

#     if type == "fourier":
#         start = fourier_start
#         stop = fourier_stop
#     else:
#         start = legendre_start
#         stop = legendre_stop

#     for l in range(start, stop):
#         means.append(np.mean(data[l]))
#         deviations.append(np.std(data[l]))

#     means = np.asarray(means)
#     deviations = np.asarray(deviations)

#     chi_min = 10000000
#     s_min = 0

#     for i in range(steps):

#         chi_square = chi_square_func(s, means, deviations, type)
#         chi_history.append(chi_square)

#         grad = 0
#         for l in range(start, stop):
#             grad += (
#                 2
#                 * (q(l) / p(l))
#                 * (
#                     r(s, means, deviations, type)
#                     / (((p(l) * deviations[l - start]) * (1 + s * q(l) / p(l))) ** 2)
#                 )
#                 * (
#                     p(l) * means[l - start]
#                     - (r(s, means, deviations, type) / (1 + s * q(l) / p(l)))
#                 )
#             )

#         s -= 0.1 * grad

#         if chi_square_func(s, means, deviations, type) < chi_min:
#             chi_min = chi_square_func(s, means, deviations, type)
#             s_min = s
#             arg = i

#     return (s_min, chi_min, means, deviations, chi_history, arg)

# reconstruction of chi square numerically
def chi_plot(data, type):

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

        deviations.append(np.std(data[l]) / np.sqrt(len(data)))

    means = np.asarray(means)
    deviations = np.asarray(deviations)

    chi_history = []
    s = np.linspace(-10, 80, 10000)
    for i in range(len(s)):
        chi_history.append(chi_square_func(s[i], means, deviations, type))

    fig, ax = plt.subplots()
    ax.plot(s, chi_history, color="black", lw=2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title(r"$\chi^2$ vs. s")
    ax.set_ylabel(r"$\chi^2$", fontsize=15)
    ax.set_xlabel(r"s", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(f"plots/chi_s_{type}.png", dpi=300)
    plt.show()

    chi_history = np.asarray(chi_history)
    s_min = s[np.argmin(chi_history)]

    return s_min, np.min(chi_history), means, deviations


def chi_error(s, means, deviations, type):

    if type == "fourier":
        start = fourier_start
        stop = fourier_stop
    else:
        start = legendre_start
        stop = legendre_stop

    H_11 = 0
    H_22 = 0
    H_12 = 0

    for l in range(start, stop):
        H_11 += 2 / ((p(l) * deviations[l - start]) * (1 + ((s * q(l)) / p(l)))) ** 2
        H_12 += (
            (2 * q(l) / p(l))
            * (1 / (((p(l) * deviations[l - start]) * (1 + s * q(l) / p(l))) ** 2))
            * (
                p(l) * means[l - start]
                - 2 * (r(s, means, deviations, type) / (1 + s * q(l) / p(l)))
            )
        )
        H_22 += (
            2
            * ((p(l) / p(l)) ** 2)
            * (1 / ((p(l) * deviations[l - start]) ** 2))
            * (1 / (1 + s * q(l) / p(l)) ** 3)
            * (
                (3 * r(s, means, deviations, type) / (1 + s * q(l) / p(l)))
                - 2 * p(l) * means[l - start]
            )
        )

    return np.array([[H_11, H_12], [H_12, H_22]])


# Fourier Analysis
plt.rcParams["figure.figsize"] = (10, 3)
# fourier amplitude data
a = extract("fourier_amplitudes.txt")
a, thres_dict = hard_threshold(a, "fourier")
plt.rcParams["axes.facecolor"] = "grey"
plt.rcParams["figure.figsize"] = (12, 4)
a = np.copy(soft_threshold(a, "fourier", thres_dict))
a = a.T

with open("build/fourier_data.txt", "w") as f:
    for line in a:
        for element in line:
            f.write(str(element))
            f.write(",")
        f.write("\n")
f.close()

print("\nProcessing...", end="\r")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (8, 6)
s, chi_min, means, deviations = chi_plot(a, "fourier")
kappa = 1 / r(s, means, deviations, "fourier")
sigma = s

print("\nFourier Analysis Results:")
error_hessian = chi_error(s, means, deviations, "fourier")
print("\nH11 = ", np.format_float_scientific(error_hessian[0, 0], precision=4))
print("\nH12 = ", np.format_float_scientific(error_hessian[0, 1], precision=4))
print("\nH22 = ", np.format_float_scientific(error_hessian[1, 1], precision=4))
inv_hess = np.linalg.inv(error_hessian)
print("\nH'11 = ", np.format_float_scientific(inv_hess[0, 0], precision=4))
print("\nH'12 = ", np.format_float_scientific(inv_hess[0, 1], precision=4))
print("\nH'22 = ", np.format_float_scientific(inv_hess[1, 1], precision=4))
kappa_err = np.sqrt(
    (inv_hess[0, 0] * kappa**2) / r(s, means, deviations, "fourier") ** 2
)
sigma_err = np.sqrt(inv_hess[1, 1])

print("\nkappa = {:.4f}".format(kappa), "\u00B1", "{:.4f}".format(kappa_err))
print("sigma = {:.4f}".format(sigma), "\u00B1", "{:.4f}".format(sigma_err))
print("minimum chi_sq = {:.4f}".format(chi_min), "\n")

plt.rcParams["figure.figsize"] = (12, 4)
# theoretical vs. experimental fourier amplitude
a_t = []
for l in range(fourier_start, fourier_stop):
    a_t.append(
        (2 * l + 1) / (4 * np.pi * kappa * (l - 1) * (l + 2) * (l * (l + 1) + sigma))
    )
# plt.plot(range(fourier_start, fourier_stop), a_t, color="red", label="theoretical")
plt.errorbar(
    range(fourier_start, fourier_stop),
    means,
    deviations,
    ecolor="black",
    color="blue",
    label="experimental",
    capsize=3,
)
# plt.title("Fourier Amplitude verification")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.ylabel(r"$a_k$", fontsize=15)
plt.xlabel(r"$k$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/fourier_verification_1.png", dpi=300)
plt.show()

plt.plot(
    range(fourier_start, fourier_stop), a_t, color="black", label="theoretical", ls="--"
)
plt.plot(
    range(fourier_start, fourier_stop),
    means,
    color="black",
    label="experimental",
)
# plt.title("Fourier Amplitude verification")
plt.ylabel(r"$\langle a_k \rangle$", fontsize=15)
plt.xlabel(r"$k$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/fourier_verification_2.png", dpi=300)
plt.show()

# Legendre Analysis

# legendre amplitude data
b = extract("legendre_amplitudes.txt")
b, thres_dict = hard_threshold(b, "legendre")
plt.rcParams["axes.facecolor"] = "grey"
plt.rcParams["figure.figsize"] = (12, 4)
b = np.copy(soft_threshold(b, "legendre", thres_dict))
b = b.T

with open("build/legendre_data.txt", "w") as f:
    for line in b:
        for element in line:
            f.write(str(element))
            f.write(",")
        f.write("\n")
f.close()

print("\nProcessing...", end="\r")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (8, 6)
s, chi_min, means, deviations = chi_plot(b, "legendre")
kappa = 1 / r(s, means, deviations, "legendre")
sigma = s

print("\nLegendre Analysis Results:")
error_hessian = chi_error(s, means, deviations, "legendre")
print("\nH11 = ", np.format_float_scientific(error_hessian[0, 0], precision=4))
print("\nH12 = ", np.format_float_scientific(error_hessian[0, 1], precision=4))
print("\nH22 = ", np.format_float_scientific(error_hessian[1, 1], precision=4))
inv_hess = np.linalg.inv(error_hessian)
print("\nH'11 = ", np.format_float_scientific(inv_hess[0, 0], precision=4))
print("\nH'12 = ", np.format_float_scientific(inv_hess[0, 1], precision=4))
print("\nH'22 = ", np.format_float_scientific(inv_hess[1, 1], precision=4))
kappa_err = np.sqrt(
    (inv_hess[0, 0] * kappa**2) / r(s, means, deviations, "legendre") ** 2
)
sigma_err = np.sqrt(inv_hess[1, 1])

print("\nkappa = {:.4f}".format(kappa), "\u00B1", "{:.4f}".format(kappa_err))
print("sigma = {:.4f}".format(sigma), "\u00B1", "{:.4f}".format(sigma_err))
print("minimum chi_sq = {:.4f}".format(chi_min), "\n")
plt.rcParams["figure.figsize"] = (12, 4)
# theoretical vs. experimental legendre amplitude
b_t = []
for l in range(legendre_start, legendre_stop):
    b_t.append(
        (2 * l + 1) / (4 * np.pi * kappa * (l - 1) * (l + 2) * (l * (l + 1) + sigma))
    )
# plt.plot(range(legendre_start, legendre_stop), b_t, color="red", label="theoretical")
plt.errorbar(
    range(legendre_start, legendre_stop),
    means,
    deviations,
    ecolor="black",
    color="blue",
    label="experimental",
    capsize=3,
)
# plt.title("Legendre Amplitude verification")
plt.ylabel(r"$\langle b_l \rangle$", fontsize=15)
plt.xlabel(r"$l$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/legendre_verification_1.png", dpi=300)
plt.show()

plt.plot(
    range(legendre_start, legendre_stop),
    b_t,
    color="black",
    label="theoretical",
    ls="--",
)
plt.plot(
    range(legendre_start, legendre_stop),
    means,
    color="black",
    label="experimental",
)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.title("Legendre Amplitude verification")
plt.ylabel(r"$\langle b_l \rangle$", fontsize=15)
plt.xlabel(r"$l$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/legendre_verification_2.png", dpi=300)
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
    plt.plot(np.arange(0, 60) / fps, time[l, :60] / time[l, 0], label=rf"$l={l}$")
plt.legend(bbox_to_anchor=(1.15, 1.05))
# plt.title("Time Autocorrelation")
plt.ylabel(r"C(t)", fontsize=15)
plt.xlabel(r"time (s)", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/time_autocorrelation.png")
plt.show()

for l in range(legendre_start, legendre_stop):
    plt.plot(
        np.arange(0, len(time[l, :])) / fps, time[l, :] / time[l, 0], label=rf"$l={l}$"
    )
plt.legend(bbox_to_anchor=(1.15, 1.05))
# plt.title("Time Autocorrelation")
plt.ylabel(r"C(t)", fontsize=15)
plt.xlabel(r"time (s)", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/time_autocorrelation_2.png")
plt.show()

for l in range(legendre_start, legendre_stop):
    plt.plot(np.arange(0, 60) / fps, np.log(time[l, :60]), label=rf"$l={l}$")
plt.legend(bbox_to_anchor=(1.15, 1.05))
# plt.title("Time Autocorrelation")
plt.ylabel(r"log(C(t))", fontsize=15)
plt.xlabel(r"time (s)", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
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
    plt.plot(
        np.arange(0, 60) / fps,
        time[l, :60] / time[l, 0],
        label=rf"$l={int(modes[l][0]),int(modes[l][1])}$",
    )
plt.legend(bbox_to_anchor=(1.18, 1.05))
# plt.title("Time Crosscorrelation")
plt.ylabel(r"C(t)", fontsize=15)
plt.xlabel(r"time (s)", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/time_crosscorrelation.png")
plt.show()

for l in range(len(time)):
    plt.plot(
        np.arange(0, 60) / fps,
        np.log(time[l, :60]),
        label=rf"$l={int(modes[l][0]),int(modes[l][1])}$",
    )
plt.legend(bbox_to_anchor=(1.18, 1.05))
# plt.title("Time Crosscorrelation")
plt.ylabel(r"log(C(t))", fontsize=15)
plt.xlabel(r"time (t)", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/time_crosscorrelation_log.png")
plt.show()
