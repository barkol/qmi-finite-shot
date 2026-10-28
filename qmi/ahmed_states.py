"""Zestaw rozszerzony wg Ahmed i in. (Phys. Rev. Research 3, 033278; kod: github.com/quantshah/qst-nn, MIT).

Siedem klas klasyfikacji: fock, coherent, thermal, num, gkp, binomial, cat (bez prozni; 'random' tylko w rekonstrukcji).
Definicje i zakresy parametrow wg wydanego kodu (qst_nn/ops.py), generowane przy ich odcieciu Hilberta 32 (tak jak
w ich danych) i wstawiane w nasza przestrzen 64 (CUTOFF), zeby kanaly szumu i obrazy W/Q byly identyczne jak w zestawie
podstawowym. Rozbieznosci kod/artykul: cat |alpha| w kodzie U[2,3] (artykul [1,3]); thermal nbar w kodzie (1,16]
(artykul [0,16]); przyjmujemy KOD jako zrodlo danych.
"""
import numpy as np
import qutip as qt
from scipy.special import binom

AHM_LABELS = ("fock", "coherent", "thermal", "num", "gkp", "binomial", "cat")
NA = 32            # odciecie Hilberta Ahmeda


def _rand_alpha(rng, lo, hi):
    r = rng.uniform(lo, hi); phi = rng.uniform(-np.pi, np.pi)
    return r * np.exp(1j * phi), float(r), float(phi)


def _num_codes():
    s17 = [[np.sqrt(7 - np.sqrt(17)) / np.sqrt(6), 0, 0, np.sqrt(np.sqrt(17) - 1) / np.sqrt(6), 0],
           [0, np.sqrt(9 - np.sqrt(17)) / np.sqrt(6), 0, 0, np.sqrt(np.sqrt(17) - 3) / np.sqrt(6)]]
    sM = [[0.5458351325482939, -3.7726009161224436e-9, 4.849511177634774e-8, -0.7114411727633639, -7.48481181758003e-8,
           -1.3146003192319789e-8, 0.44172510726665587, 1.1545802803733896e-8, 1.0609402576342428e-8,
           -0.028182506843720707, -6.0233214626778965e-9, -6.392041552216322e-9, 0.00037641909140801935,
           -6.9186916801058116e-9],
          [2.48926815257019e-9, -0.7446851186077535, -8.040831059521339e-9, 6.01942995399906e-8, -0.5706020908811399,
           -3.151900508005823e-8, -7.384935824733578e-10, -0.3460030551087218, -8.485651303145757e-9,
           -1.2114327561832047e-8, 0.011798401879159238, -4.660460771433317e-9, -5.090374160706911e-9,
           -0.00010758601713550998]]
    sP = [[0., 0.7562859301326029, 0., 0., -0.5151947804474741, -0.20807866860791188, 0.12704803323656158,
           0.05101928893751686, 0.3171198939841734],
          [-0.5583217426728544, -0.0020589109231194413, 0., -0.7014041964402703, -0.05583041652626998,
           0.0005664728465725445, -0.2755044401850055, -0.3333309025086189, 0.0785824556163142]]
    sP2 = [[-0.5046617350158988, 0.08380989527942606, -0.225295417417812, 0., -0.45359477373452817,
            -0.5236866813756252, 0.2523308675079494, 0., 0.09562538828178244, 0.2172849136874009, 0., 0., 0.,
            -0.2793663175980869, -0.08280858231312467, -0.05106696128137072],
           [-0.0014249418817930378, 0.5018692341095683, 0.4839749920101922, -0.3874886488913531, 0.055390715144453026,
            -0.25780190053922486, -0.08970154713375252, -0.1892386424818236, 0.10840637100094529,
            -0.19963901508324772, -0.41852779130900664, -0.05747247660559087, 0., -0.0007888071131354318,
            -0.1424131123943283, -0.0001441905475623907]]
    return [s17, sM, _sM2(), sP, sP2]


def _sM2():
    c = complex
    a = [-0.45717455741713664, c(-1.0856965103853774e-6, 1.3239037829080093e-6), c(-0.35772784377291084, -0.048007740168066144),
         c(-3.5459165445315755e-6, 0.000012571453643232864), c(-0.5383420820794502, -0.24179040513272307),
         c(9.675641330014822e-7, 4.569566899500361e-6), c(0.2587482691377581, 0.313044506480362),
         c(4.1979351791851435e-6, -1.122460690803522e-6), c(-0.11094500303308243, 0.20905585817734396),
         c(-1.1837814323046472e-6, 3.8758497675466054e-7), c(0.1275629945870373, -0.1177987279989385),
         c(-2.690647673469878e-6, -3.6519804939862998e-6), c(0.12095531973074151, -0.19588735180644176),
         c(-2.6588791126371675e-6, -6.058292629669095e-7), c(0.052905370429015865, -0.0626791930782206),
         c(-1.6615538648519722e-7, 6.756126951837809e-8), c(0.016378329200891946, -0.034743342821208854),
         c(4.408946495377283e-8, 2.2826415255126898e-8), c(0.002765352838800482, -0.010624191776867055),
         6.429253878486627e-8, c(0.00027095836439738105, -0.002684435917226972),
         c(1.1081202749445256e-8, -2.938812506852636e-8), c(-0.000055767533641099717, -0.000525444354381421),
         c(-1.0776974926155464e-8, -2.497769263148397e-8), c(-0.000024992489351114305, -0.00008178444317382933),
         c(-1.5079116121444066e-8, -2.0513760149701907e-8), c(-5.64035228941742e-6, -0.000010297667130821428),
         c(-1.488452012610573e-8, -1.7358623165948514e-8), c(-8.909884885392901e-7, -1.04267002748775e-6),
         c(-1.2056784102984098e-8, -1.2210951690230782e-8)]
    b = [0, 0.5871298855433338, c(-3.3729618710801137e-6, 2.4152360811650373e-6),
         c(-0.5233926069798007, -0.13655786303346068), c(-4.623380373113224e-6, 0.000010362902695259763),
         c(-0.17909656013941788, -0.11916639160269833), c(-3.399720873431807e-6, -7.125008373682292e-7),
         c(0.04072119358712736, -0.3719310475303641), c(-7.536125619789242e-6, 1.885248226837573e-6),
         c(-0.11393851510585044, -0.3456924286310791), c(-2.3915763815197452e-6, -4.2406689395594674e-7),
         c(0.12820184730203607, 0.0935942533049232), c(-1.5407293261691393e-6, -2.4673669087089514e-6),
         c(-0.012272903377715643, -0.13317144020065683), c(-1.1260776123106269e-6, -1.6865728072273087e-7),
         c(-0.01013345155253134, -0.0240812705564227), c(0., -1.4163391111474348e-7),
         c(-0.003213070562510137, -0.012363639898516247), c(-1.0619280312362908e-8, -1.2021213613319027e-7),
         c(-0.002006756716685063, -0.0026636832583059812), c(0., -4.509035934797572e-8),
         c(-0.00048585160444833446, -0.0005014735884977489), c(-1.2286988061034212e-8, -2.1199721851825594e-8),
         c(-0.00010897007463988193, -0.00007018240288615613), c(-1.2811279935244964e-8, -1.160553871672415e-8),
         c(-0.00001785800494916693, -6.603027186486886e-6), -1.1639448324793031e-8,
         c(-2.4097385882316104e-6, -3.5223103057306496e-7), -1.0792272866841885e-8,
         c(-2.597671478115077e-7, 2.622928060603902e-8)]
    return [a, b]


_NUM = None


def _embed(rho, N):
    """rho (NA x NA, Qobj) -> Qobj o wymiarze N."""
    m = np.zeros((N, N), dtype=complex); m[:NA, :NA] = rho.full()
    return qt.Qobj(m, dims=[[N], [N]])


def sample_ahmed(label, N, rng):
    global _NUM
    h = NA
    if label == "fock":
        n = int(rng.integers(1, h // 2 + 1)); rho = qt.fock_dm(h, n); st = {"n": n}
    elif label == "coherent":
        a, r, ph = _rand_alpha(rng, 1e-6, 3.0); rho = qt.coherent_dm(h, a); st = {"amp": r, "phase": ph}
    elif label == "thermal":
        nb = float(rng.uniform(1.0, h / 2)); rho = qt.thermal_dm(h, nb); st = {"nbar": nb}
    elif label == "num":
        if _NUM is None: _NUM = _num_codes()
        k = int(rng.integers(len(_NUM))); mu = int(rng.integers(2)); p = _NUM[k][mu]
        ket = qt.Qobj(np.zeros((h, 1), dtype=complex))
        for n, c in enumerate(p): ket += c * qt.basis(h, n)
        rho = qt.ket2dm(ket.unit()); st = {"code": k, "mu": mu}
    elif label == "gkp":
        mu = int(rng.integers(2)); delta = float(rng.uniform(0.2, 0.5)); c = np.sqrt(np.pi / 2)
        ket = qt.Qobj(np.zeros((h, 1), dtype=complex))
        for n1 in range(-20, 20):
            for n2 in range(-20, 20):
                a = c * (2 * n1 + mu + 1j * n2)
                w = np.exp(-delta ** 2 * abs(a) ** 2)
                if w < 1e-12: continue
                ket += w * np.exp(-1j * c ** 2 * 2 * n1 * n2) * qt.coherent(h, a)
        rho = qt.ket2dm(ket.unit()); st = {"delta": delta, "mu": mu}
    elif label == "binomial":
        S = int(rng.integers(1, 10)); nmax = int(h / (S + 1)) - 1
        Nb = int(rng.integers(2, nmax)) if nmax > 2 else nmax
        mu = int(rng.integers(2)); ket = qt.Qobj(np.zeros((h, 1), dtype=complex))
        for m in range(Nb):
            ket += ((-1) ** (mu * m)) * np.sqrt(binom(Nb + 1, m)) * qt.fock(h, (S + 1) * m)
        rho = qt.ket2dm(ket.unit()); st = {"S": S, "N": Nb, "mu": mu}
    elif label == "cat":
        a0, r, ph = _rand_alpha(rng, 2.0, 3.0)      # kod: random_alpha(2, 3) -> |alpha| in [2,3]
        S = int(rng.integers(0, 3)); mu = int(rng.integers(2)); ket = 0 * qt.coherent(h, 0)
        for k in range(0, int((2 * S + 2) / 2)):
            sign = (-1) ** int(mu > 0.5) if k >= S else 1
            pre = np.exp(1j * (np.pi / (S + 1)) * k); al = pre * a0 * (-(1j ** mu))
            ket += sign * qt.coherent(h, al) + sign * qt.coherent(h, -al)
        rho = qt.ket2dm(ket.unit()); st = {"amp": r, "phase": ph, "S": S, "mu": mu}
    else:
        raise ValueError(label)
    return _embed(rho, N), st


def in_holdout_ahm(label, st):
    """Hold-out dla zestawu Ahmeda: gorny skraj zakresu parametru kazdej rodziny (ok. 1/8 zakresu)."""
    if label == "fock":     return st["n"] in (15, 16)          # n w 1..16
    if label == "coherent": return st["amp"] >= 2.6             # |alpha| w (0,3]
    if label == "thermal":  return st["nbar"] >= 14.0           # nbar w [1,16]
    if label == "num":      return st["code"] == 4              # 5 kodow numerycznych, ostatni wylaczony
    if label == "gkp":      return st["delta"] >= 0.46          # delta w [0.2,0.5]
    if label == "binomial": return st["S"] in (8, 9)            # S w 1..9
    if label == "cat":      return st["amp"] >= 2.85            # |alpha| w [2,3]
    raise ValueError(label)


def make_sampler_ahm(mode):
    """mode: none | train (bez hold-outu) | test (tylko hold-out)."""
    if mode == "none": return sample_ahmed
    def sampler(label, N, rng):
        while True:
            rho, st = sample_ahmed(label, N, rng); h = in_holdout_ahm(label, st)
            if (mode == "train" and not h) or (mode == "test" and h): return rho, st
    return sampler
