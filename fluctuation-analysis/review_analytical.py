import numpy as np
import matplotlib.pyplot as plt

fourier_start = int(input("\nEnter first mode to be analyzed for Fourier: "))
fourier_stop = int(input("\nEnter last mode to be analyzed for Fourier: ")) + 1

legendre_start = int(input("\nEnter first mode to be analyzed for Legendre: "))
legendre_stop = int(input("\nEnter last mode to be analyzed for Legendre: ")) + 1


def extract(filename):
    data = []
    with open(f"build/{filename}", "r") as f:
        data = f.readlines()
    f.close()

    data_new = []
    for i in range(len(data)):
        if len(data[i].split(",")[:-1]) == len(data[1].split(",")[:-1]):
            data_new.append(data[i].split(",")[:-1])

    return np.asarray(data_new).astype("float64")


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

    return s_min, np.min(chi_history), means, deviations, chi_history


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
a = extract("fourier_data.txt")
plt.rcParams["axes.facecolor"] = "grey"
# plt.rcParams["savefig.facecolor"] = "grey"
plt.rcParams["figure.figsize"] = (12, 4)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# for l in range(fourier_start, fourier_stop):
#     fig, ax = plt.subplots()
#     ax.plot(
#         np.arange(0, len(a[l])) / 29,
#         a[l],
#         label=rf"$k = {l}$",
#         color="white",
#     )
#     label_mean = round(np.mean(a[l]), 7)
#     ax.plot(
#         np.arange(0, len(a[l])) / 29,
#         (np.zeros(len(a[l])) + (np.mean(a[l]))),
#         color="black",
#         label=rf"$<a> = {label_mean}$",
#         lw=2,
#     )
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     ax.set_ylabel(r"$a_k$", fontsize=15)
#     ax.set_xlabel(r"time (s)", fontsize=15)
#     ax.tick_params(axis="both", labelsize=12)
#     ax.legend(loc="upper left")
#     fig.tight_layout()
#     fig.savefig(f"plots/fourier_threshold_{l}.png", dpi=300)
#     plt.show()

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (8, 6)
s, chi_min, means, deviations, chi_history = chi_plot(a, "fourier")
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
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.title("Fourier Amplitude verification")
plt.ylabel(r"$<a_k>$", fontsize=15)
plt.xlabel(r"$k$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/fourier_verification_1.png", dpi=300)
plt.show()

plt.plot(
    range(fourier_start, fourier_stop),
    a_t,
    color="black",
    label="theoretical",
    ls="--",
)
plt.plot(
    range(fourier_start, fourier_stop),
    means,
    color="black",
    label="experimental",
)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.title("Legendre Amplitude verification")
plt.ylabel(r"$<a_l>$", fontsize=15)
plt.xlabel(r"$k$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/fourier_verification_2.png", dpi=300)
plt.show()

# s_arr = np.linspace(-10, 80, 10000)
# for i in range(-5, 6):
#     print(
#         chi_square_func(
#             (s_arr[np.where(s_arr == sigma)[0] + i]),
#             means,
#             deviations,
#             "fourier",
#         ),
#         s_arr[np.where(s_arr == sigma)[0] + i],
#     )

# print(
#     "\n",
#     (
#         chi_square_func(sigma + 0.009, means, deviations, "fourier")
#         - 2 * chi_square_func(sigma, means, deviations, "fourier")
#         + chi_square_func(sigma - 0.009, means, deviations, "fourier")
#     )
#     / 0.009**2,
# )

# Legendre Analysis

b = extract("legendre_data.txt")

plt.rcParams["axes.facecolor"] = "grey"
# plt.rcParams["savefig.facecolor"] = "grey"
plt.rcParams["figure.figsize"] = (12, 4)
# for l in range(legendre_start, legendre_stop):
#     plt.plot(
#         np.arange(0, len(b[l])) / 29,
#         b[l],
#         label=rf"$l = {l}$",
#         color="white",
#     )
#     label_mean = round(np.mean(b[l]), 7)
#     plt.plot(
#         np.arange(0, len(a[l])) / 29,
#         (np.zeros(len(a[l])) + (np.mean(b[l]))),
#         color="black",
#         label=rf"$<b> = {label_mean}$",
#         lw=2,
#     )
#     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     plt.ylabel(r"$b_l$", fontsize=15)
#     plt.xlabel(r"time (s)", fontsize=15)
#     plt.yticks(fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.legend(loc="upper left")
#     plt.tight_layout()
#     plt.savefig(f"plots/legendre_threshold_{l}.png", dpi=300)
#     plt.show()

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (8, 6)
s, chi_min, means, deviations, chi_history = chi_plot(b, "legendre")
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
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.title("Legendre Amplitude verification")
plt.ylabel(r"$<b_l>$", fontsize=15)
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
plt.ylabel(r"$<b_l>$", fontsize=15)
plt.xlabel(r"$l$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plots/legendre_verification_2.png", dpi=300)
plt.show()

# for i in range(-5, 6):
#     print(
#         chi_square_func(
#             (s_arr[np.where(s_arr == sigma)[0] + i]),
#             means,
#             deviations,
#             "legendre",
#         ),
#         s_arr[np.where(s_arr == sigma)[0] + i],
#     )

# print(
#     "\n",
#     (
#         chi_square_func(sigma + 0.009, means, deviations, "legendre")
#         - 2 * chi_square_func(sigma, means, deviations, "legendre")
#         + chi_square_func(sigma - 0.009, means, deviations, "legendre")
#     )
#     / 0.009**2,
# )


# todo check second derivative
