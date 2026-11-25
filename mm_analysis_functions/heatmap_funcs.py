import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.special import expit


class aniso_distribution:
    def __init__(self, weights, centres):
        self.weights = np.asarray(weights, dtype = float)
        self.centres = np.asarray(centres, dtype=float)

    def S(self):
        mask = self.centres > 0
        return np.sum(self.weights[mask] * self.centres[mask])
    def total(self):
        return np.sum(self.weights)
    def mean(self):
        return np.average(self.centres, weights=self.weights)
    def median(self):
        order = np.argsort(self.centres)
        centres = self.centres[order]
        weights = self.weights[order]
        cumulative = np.cumsum(weights)
        return centres[np.searchsorted(cumulative, 0.5*cumulative[-1])]
    def mode(self):
        return self.centres[np.argmax(self.weights)]
    def second_moment(self):
        return np.average(self.centres**2, weights=self.weights)
    def third_moment(self):
        return np.average(self.centres**3, weights=self.weights)
    def fourth_moment(self):
        return np.average(self.centres**4, weights=self.weights)
    def std(self):
        return np.sqrt(self.second_moment() - self.mean()**2)
    def positive_fraction(self):
        return np.sum( self.weights * (self.centres > 0) ) / self.total()
    def mod_ratio(self):
        positive = np.sum( self.weights * np.maximum(0, self.centres))
        negative = np.sum( self.weights * np.minimum(0, self.centres))
        if negative != 0:
            return abs(positive / negative)
        else:
            return np.nan
    def binder(self):
        B = 1 - self.fourth_moment() / (3 * (self.second_moment())**2)
        return B
    def pearson_skew(self):
        S = 3 * (self.mean() - self.median()) / max(self.std(), 1e-6)
        return S


    def measures(self):
        return {
            "mean": self.mean(),
            "second_moment": self.second_moment(),
            "third_moment": self.third_moment(),
            "fourth_moment": self.fourth_moment(),
            "binder": self.binder(),
            "S": self.S(),
            "positive_fraction": self.positive_fraction(),
            "std": self.std(),
            "mod_ratio": self.mod_ratio(),
            "pearson_skew": self.pearson_skew()
            }


def load_anisos(A_file_names, D_file_names, Dt_set, ws, L, occupancy_cutoff):
    anisos = [[[] for _ in ws] for _ in Dt_set]
    all_occ = [[[] for _ in ws] for _ in Dt_set]
    areas = (L * np.array(ws) / ws[-1])**2
    for Afile, Dfile in zip(A_file_names, D_file_names):
        Adata = np.load(Afile, allow_pickle=True)
        Ddata = np.load(Dfile, allow_pickle=True)

        for dt_idx in range(len(Dt_set)):
            for episode_idx in range(len(Adata[dt_idx])):
                for ws_idx in range(len(ws)):

                    A = np.asarray(Adata[dt_idx][episode_idx][ws_idx]).ravel()
                    D = np.asarray(Ddata[dt_idx][episode_idx][ws_idx]).ravel()

                    valid = np.isfinite(A) & np.isfinite(D)
                    A = A[valid]
                    D = D[valid]
                    occ = D * areas[ws_idx]
                    mask = (occ >= occupancy_cutoff)
                    anisos[dt_idx][ws_idx].extend(A[mask])
                    all_occ[dt_idx][ws_idx].extend(occ[mask])      
    return anisos

def make_heatmaps(anisos, Dt_set, ws):
    dists = [[None for _ in ws] for _ in Dt_set]
    all_heatmaps = {}
    bins = np.linspace(-1,1,101)
    centres = (bins[:-1] + bins[1:]) / 2
    for dt_idx in range(len(Dt_set)):
        for ws_idx in range(len(ws)):
            A = np.array(anisos[dt_idx][ws_idx])
            counts, _ = np.histogram(A, bins=bins)
            weights = counts / counts.sum()
            dists[dt_idx][ws_idx] = aniso_distribution(weights, centres)
            for measure in dists[dt_idx][ws_idx].measures():
                if measure not in all_heatmaps:
                    all_heatmaps[measure] = np.full((len(Dt_set), len(ws)), np.nan)
                all_heatmaps[measure][dt_idx][ws_idx] = dists[dt_idx][ws_idx].measures()[measure]
    return all_heatmaps


"""
FIND INFLECTION/ PEAK OF a heat slice
"""

def sigmoid(x, A, x0, k, c):
    return (A / (1 + expit(-(x0 - x) / k))) + c

def find_peak_and_inflect(heats_slice, tile_sizes):
    y = np.array(heats_slice)
    x = np.log(np.array(tile_sizes))

    if len(y) == 0:
        return np.nan, np.nan

    # SPLINE PEAKS
    peaks, _ = find_peaks(y, prominence=0.05)
    if len(peaks):
        peak_idx = peaks[np.argmax(y[peaks])]
        left = max(0, peak_idx - 2)
        right = min(len(x), peak_idx + 3)

        sp = UnivariateSpline(x[left:right], y[left:right], s=1e-3)
        xp = np.linspace(x[left], x[right - 1], 500)

        xpk = xp[np.argmax(sp(xp))]
        wpk = np.exp(xpk)
        ypk = sp(xpk)
    else:
        wpk, ypk = tile_sizes[np.argmax(y)], max(y)
        xpk = np.log(wpk)
    slice_peak = wpk

    # FIT INFLECT RIGHT OF PEAK
    p = np.argmin(np.abs(x - xpk))
    xfit, yfit = x[p:], y[p:]
    if len(xfit) > 1:
        eps = 1e-6
        try:
            popt, _ = curve_fit(
                sigmoid,
                xfit,
                yfit,
                p0=[yfit[0], np.median(xfit), 0.5, 0],
                bounds=([-np.inf, xfit[0], 1e-3, -np.inf], [np.inf, xfit[-1], np.inf, np.inf])
            )
            A_fit, x0, k, c = popt
            w0 = np.exp(x0)
        except:
            w0 = np.nan
            popt = None
    else:
        w0 = np.nan
        popt = None
    slice_inflect = w0
    return slice_peak, slice_inflect, popt