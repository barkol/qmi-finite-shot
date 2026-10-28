# Rekordy jednego modu stanu EPR (Kawasaki i in., Dryad doi:10.5061/dryad.dbrv15fbq) a klasyfikacja qsdl-qmi

Wszystkie skrypty, logi i wyniki są w `analyses/kawasaki_epr/` (duże pliki: MANIFEST_LARGE_FILES.md). Liczby zbiorczo: `wyniki.json`.

## 0. Źródło i uwaga o tytule

Zbiór na Dryad nosi tytuł „...using optical parametric **amplifiers**” (nie „oscillators”). Jest to zbiór danych do pracy A. Kawasaki i in., *Real-time observation of picosecond-timescale optical quantum entanglement toward ultrafast quantum information processing*, arXiv:2403.07357 (v1, 12.03.2024), opublikowanej w Nature Photonics (DOI 10.1038/s41566-024-01589-7, styczeń 2025). Kopia PDF: `zrodla/arxiv2403.07357.pdf`. Układ: dwa źródła ściśnięcia na falowodach PPLN (OPA) łączone na płytce 50:50 dają stan EPR, który mierzą dwa homodyny 70 GHz poprzedzone wzmacniaczami fazoczułymi (OPA pomiarowe, około 25 dB). Oscyloskop: 256 GSa/s, 5000 ramek po 5121 próbek. Ramki są zapisywane co 400 µs, w 40-µs oknach pomiarowych cyklu blokady fazy.

Praca nie podaje funkcji modu wprost. Odsyła do suplementu (niedostępnego na arXiv) i stwierdza, że kształt pakietu jest „the same shape as one employed in conventional quantum entanglement observations [11]” i że „only the time width is shortened by a factor of 10^3 from ... 40 ns [11]”. Ref. [11] to Asavanant i in., Science 366, 373 (2019). W ich suplemencie (`zrodla/asavanant1903.03918.pdf`, równanie S1) funkcja ma postać f(t) ∝ (t − t0 + 0,5 + tc) exp[−γ²(t − t0 + 0,5)²] w oknie |t − t0| ≤ Δt/2, z Δt = 40 ns, γ = 2π × 10,5 MHz i tc = 0.

## 1. Przetwarzanie (krok12_kwadratury.py, wspolne.py)

- Format: Keysight H5, kanał jednokanałowy, int16 × YInc + YOrg [V]. Strażnik sprawdza XInc = 3,90625 ps, 5121 próbek i 5000 segmentów. Znaczniki czasu segmentów hd1 i hd2 są identyczne, więc oba kanały rejestrowano jednocześnie.
- Mod czasowy: S1 przeskalowane o 10³, czyli Δt = 40 ps (10 próbek, dokładnie 39,06 ps), γ = 2π × 10,5 GHz. Pakiety są rozłączne i znormalizowane (Σf² = 1). Funkcja jest antysymetryczna, więc wycina składową stałą. Jedno uproszczenie: pół próbki przesunięcia z S1 (0,5 ns przy 1 GSa/s) zastąpiłem środkiem okna o parzystej liczbie próbek. Wynik jest mało czuły na γ (7 i 10,5 GHz dają te same wariancje z dokładnością 0,01 dB), a pakiet 20-próbkowy zmienia wariancje o mniej niż 0,15 dB.
- Opóźnienie kanałów: bez wyrównania x1 i x2 nie są skorelowane. Korelacja wzajemna ma ostre maksimum przy hd2 opóźnionym o **308 próbek (1,20 ns)**, z korelacją 0,88 w x i 0,89 w p. To samo opóźnienie wychodzi we wszystkich 14 zestawach. Przesunięcie o ±1 próbkę niszczy korelację EPR (z −4,45 dB do +1 dB), więc wyrównanie jest istotne i jednoznaczne.
- Kalibracja: x = q/√(2 Var_shot), więc próżnia ma Var 1/2 (konwencja projektu x = (a + a†)/√2). Var_shot to średnia z shot_before i shot_after (różnią się o ≤ 0,4 dB). Szum śrutowy jest zmierzony za wzmacniaczem fazoczułym, co wynika ze stosunków wariancji. Seria EPR1 (pompa 1: 50, 100 i 150 mW) nie ma własnych plików shot, więc użyłem shot z 200mW_800mW_00dB tej serii. Te pliki nagrano przed pierwszym (13:37) i po ostatnim (14:02) zestawie serii.
- Szum elektroniki (20230808_bEPR_circuitnoise/hd1, hd2): po tej samej funkcji modu odstęp od szumu śrutowego wynosi 15,0 dB (hd1) i 12,6–12,8 dB (hd2) przy pełnym wzmocnieniu. Spada do 2,3/1,4 dB w zestawie „20 dB” serii strat oraz do 3,5/2,4 dB przy pompie 2 = 200 mW. Dodatkowa strata i słabsze wzmocnienie obniżają wzmocniony szum śrutowy, a elektronika zostaje. **Szumu elektroniki nie odejmuję od zdarzeń.** Zawarty w Var_shot działa dokładnie jak dodatkowa strata η_el = 1 − V_el/V_shot (szum gaussowski o wariancji (1 − η)/2 jest nieodróżnialny od domieszki próżni). Rekordy są więc rekordami stanu fizycznego po tej stracie. Wartości po odjęciu elektroniki podaję osobno (indeks „el”).
- Zapis: kwadratury wszystkich pakietów (4 × 2,4 mln na zestaw, float32) w `dane/<seria>__<zestaw>.npz` (razem 514 MB) oraz metryki w `.json`. Surowe HDF5 (6,5 GB) zostały rozpakowane w `raw/bEPR_for_dryad/` zgodnie z poleceniem. Można je usunąć, bo wszystko dalej działa na npz.

## 2. Kontrole fizyczne

Wybrane zestawy (pełna tabela w `wyniki.json`, klucz `zestawy`). Var to wariancja przy próżni 1/2. „x−” oznacza 10 log₁₀[Var((x1 − x2)/√2)/(1/2)]. W p ściśnięta jest kombinacja p1 − p2 (znak LO hd2 w p jest odwrócony względem konwencji pracy, w której ściśnięte jest p+).

| zestaw (pompa1_pompa2_strata) | odstęp el. hd1/hd2 [dB] | x− [dB] (el) | p ściśn. [dB] (el) | n̄ hd1 (el) | n̄ hd2 | Vp/Vx [dB] |
|---|---|---|---|---|---|---|
| 050_800_00 | 15,0/12,8 | −3,57 (−3,84) | −3,06 (−3,27) | 0,56 (0,57) | 0,47 | 0,85 |
| 100_800_00 | 15,0/12,8 | −4,26 (−4,62) | −3,85 (−4,15) | 1,45 (1,49) | 1,27 | 1,15 |
| 150_800_00 | 15,0/12,8 | −4,14 (−4,51) | −4,32 (−4,66) | 2,69 (2,78) | 2,51 | 0,83 |
| 200_800_00 (EPR1) | 15,0/12,8 | −4,42 (−4,84) | −4,23 (−4,59) | 4,36 (4,50) | 3,99 | 0,32 |
| 200_200_00 | 3,5/2,4 | −1,12 (−3,59) | −1,13 (−3,50) | 2,05 (3,72) | 1,51 | 0,89 |
| 200_400_00 | 7,1/5,7 | −2,89 (−4,61) | −2,74 (−4,25) | 3,31 (4,12) | 2,90 | 0,92 |
| 200_600_00 | 11,2/9,0 | −3,87 (−4,75) | −3,75 (−4,46) | 3,93 (4,25) | 3,70 | 1,15 |
| 200_800_00 (EPR2) | 15,0/12,6 | −4,31 (−4,75) | −4,21 (−4,55) | 4,36 (4,51) | 4,03 | 0,66 |
| 200_800_00 (EPR3) | 15,0/12,6 | −4,45 (−4,89) | −4,26 (−4,61) | 4,11 (4,24) | 3,85 | 0,65 |
| 200_800_05 | 10,5/8,5 | −3,53 (−4,52) | −3,43 (−4,32) | 4,01 (4,40) | 3,44 | 0,56 |
| 200_800_10 | 6,9/5,0 | −2,27 (−4,09) | −2,18 (−3,86) | 3,32 (4,18) | 2,57 | 0,62 |
| 200_800_20 | 2,3/1,4 | −0,33 (−2,02) | −0,32 (−1,92) | 1,05 (2,56) | 0,62 | 0,47 |

- **Porównanie z pracą.** Przy nastawach z pracy (200 mW / 800 mW) autorzy podają w pakiecie 40 ps x− = −4,7 dB i p+ = −4,5 dB. My mamy −4,31 do −4,45 dB i −4,21 do −4,26 dB bez korekty oraz −4,75 do −4,89 dB i −4,55 do −4,61 dB po odjęciu elektroniki (trzy niezależne nagrania tych nastaw). Wartości autorów leżą między naszymi. Ich wartość dla pojedynczej próbki przy τ = 0 (4,0 dB) odpowiada naszym −3,5 dB bez korekty. Przetwarzanie odtwarza więc liczby z pracy z dokładnością około 0,3 dB, a resztę można przypisać nieznanemu szczegółowi (kształt i przesunięcie pakietu, sposób odjęcia elektroniki).
- **Stan jednego modu jest gaussowski.** Na 2,4 mln pakietów nadmiarowa kurtoza wynosi |κ| ≤ 0,04, a średnie są równe 0 z dokładnością 0,004. Wariancja rośnie z pompą 1 tak, jak dla TMSV: przy modelu czystej straty n̄ = η sinh² r, Var(x−)/V_vac = η e^{−2r} + 1 − η i r = a√P1 jedno wspólne η = 0,661 oraz a = 0,116 mW^{−1/2} opisują n̄ i ściśnięcie czterech zestawów serii EPR1 z RMS 3,5% (r = 1,64 przy 200 mW, czyli 14,2 dB ściśnięcia w samym źródle). Dla porównania praca podaje η_state η_meas = 0,94 × 0,76 = 0,71. W seriach pompy 2 i strat r zostaje stałe (1,63–1,71 z modelu, poza „20 dB”, gdzie dominuje elektronika), a zmienia się tylko η (0,24–0,66). Zgadza się to z rolą wzmacniacza pomiarowego i strat za nim.
- **Stan nie jest dokładnie termiczny.** Var p jest większe od Var x o 0,3–1,15 dB (hd1, podobnie hd2). Ślad częściowy EPR z dwóch nierównych źródeł ściśnięcia jest ściśniętym stanem termicznym, a nie stanem fazowo niezmienniczym. Asymetria jest mała względem n̄ (stopień ściśnięcia stanu zredukowanego wynosi około 0,15 przy n̄ = 4), ale istotna statystycznie.
- **Niezależność pakietów.** Sąsiednie pakiety mają C(1) = −0,08. To samo widać w czystym szumie śrutowym, więc jest to odpowiedź detektora i filtrów, a nie stanu. Dalej mamy C(2) = 0,03 i C(3–5) ≈ ±0,01, a dla m ≥ 10 |C| ≤ 0,003. Korelacja między kolejnymi ramkami wynosi |C| ≤ 0,001 (błąd standardowy 0,0006).

## 3. Rekordy jednego modu (krok3_rekordy.py, `rekordy/`)

- Rekord: hd1 (mod A), co 40. pakiet w ramce (odstęp 1,6 ns, 12 pakietów na ramkę), ramki w kolejności czasu. Daje to 60 000 zdarzeń x (kąt 0°) i 60 000 zdarzeń p (kąt 90°) na zestaw. Rekordy hd2 zapisane do kontroli.
- n̄ = (Vx + Vp)/2 − 1/2, z błędem z bootstrapu 0,004–0,019.
- Testy jak w `qmi/x8/konno.py` (funkcje `iid` i `stac` importowane bez zmian):
  - iid (χ² jednorodności 10 bloków × 20 binów, Fisher po dwóch kątach): p od 0,005 do 0,95. p < 0,05 w 2 z 14 zestawów (EPR3 200_800_00: p = 0,005; EPR2 200_800_00: p = 0,028). Stosunek odległości blok/losowy podzbiór przy N = 10⁴ wynosi 0,92–1,10.
  - stac (7 testów celowanych, R = 200 permutacji): autokorelacja i ANOVA średnich nie wykrywają niczego we wszystkich zestawach. Test Levene’a szerokości w blokach wykrywa powolną zmianę wariancji w 4 zestawach. Najsilniej w EPR2 200_800_00 (p = 0,005/0,015/0,040 dla K = 10/50/200) i EPR3 200_800_00 (p = 0,005/0,005), słabiej w EPR1 200 i EPR3 20 dB (p = 0,02–0,04). Jest to dryf wzmocnienia lub mocy LO na poziomie ułamka dB w ciągu kilkudziesięciu sekund nagrania, zgodny z różnicą shot_before/shot_after (do 0,4 dB). Ten sam typ niestacjonarności wykazały rekordy GKP Konno.
- Zakres treningowy: zbiór bazowy (eqn) ma klasę termiczną z n̄ ∈ {2, 3, 4, 5}, a zbiór Ahmeda n̄ ∈ (1, 16]. W zakresie bazowym [2, 5] jest 11 zestawów (n̄ = 2,05–4,43). W zakresie Ahmeda jest 13 (dochodzą n̄ = 1,45 i 1,05). Zestaw 050_800_00 (n̄ = 0,56) jest poza oboma i służy jako kontrola.

## 4. Klasyfikacja (krok4_klasyf.py, krok5_agreguj.py, `klasyf/`)

**Protokół** jak `konno.klasyf`, z uogólnieniem na K kątów th_k = 180k/K. Obraz W powstaje tak: histogram na pikselach siatki potoku (QMI_XMAX = 12, GRID = 128), normalizacja masy rzutu do próżni, a potem `qmi.shots.fbp`.

**Sieci.** Dla K = 6: bazowe `runs_eq/clf_W_shotsK6_s0–2` (te same co w teście GKP) oraz sieci Ahmeda K = 6 `kawasaki/runs_ahmK6/clf_W_shotsK6_s0–2`. Tych drugich brak w `runs_ahm_eq`. Wytrenowała je równoległa analiza PSS według przepisu runs_eq (`kawasaki/ahm_k6.sh`); użyłem ich tylko do odczytu. Dla K = 36: bazowe `runs_eq/clf_W_shots_s0–4` i Ahmeda `runs_ahm_eq/clf_W_shots_s0–2`. Inferencja szła na CPU.

**Fazy.** Rekord ma dwie fazy, a sieci K = 2 nie istnieją. Rozłożenie zdarzeń na K faz zakłada fazową niezmienniczość stanu. Wariant **A**: 0° ← x, 90° ← p, pozostałe kąty ← losowo z puli pozostałych zdarzeń x i p, bez powtórzeń w rekordzie. Wariant **B**: wszystkie kąty z puli x ∪ p (pełna symetryzacja). Wobec asymetrii 0,3–1,15 dB jest to przybliżenie. Odpowiada źródłu o fazie losowej względem LO, czyli uśrednionemu po fazie ściśniętemu stanowi termicznemu.

**Źródła.** exp_los to losowe podzbiory rekordu (200 losowań). exp_blok to kolejne bloki czasu. sym_gauss to to samo przypisanie, ale zdarzenia z N(0, Vx) i N(0, Vp) o zmierzonych wariancjach; jest to model szumu zliczeń, a różnica exp − sym_gauss testuje iid i gaussowskość. sym_term to stan termiczny o zmierzonym n̄, z N(0, n̄ + 1/2) na każdym kącie. Rzut Radona termicznego W z siatki potoku i rozkład gaussowski na binach różnią się o TVD ≤ 5·10⁻⁴. Dla K = 6 policzyłem warianty A i B na wszystkich 14 zestawach, z 200 losowaniami na źródło i N. Dla K = 36 (wariant B, 100 losowań) z braku czasu tylko na 5 zestawach: 050, 100, 150 i 200 mW z EPR1 oraz 200_800_10.

**Odsetek rekordów nazwanych „thermal”** (sieci uśrednione, exp_los / sym_term), K = 6, wariant A:

| zestaw | n̄ | baza N=100 | baza N=10³ | baza N=10⁴ | Ahmed N=100 | Ahmed N=10³ | Ahmed N=10⁴ |
|---|---|---|---|---|---|---|---|
| 050_800_00 (poza zakresem) | 0,56 | 0,33/0,32 | 0,38/0,39 | 0,56/0,68 | 0,15/0,15 | 0,24/0,24 | 0,39/0,52 |
| 100_800_00 | 1,45 | 0,55/0,66 | 0,92/0,91 | 1,00/0,99 | 0,24/0,29 | 0,80/0,80 | 0,99/0,99 |
| 150_800_00 | 2,69 | 0,71/0,75 | 0,98/0,99 | 1,00/1,00 | 0,36/0,38 | 0,88/0,93 | 1,00/1,00 |
| 200_800_00 (EPR1) | 4,36 | 0,75/0,78 | 0,99/1,00 | 1,00/1,00 | 0,62/0,66 | 0,95/0,95 | 1,00/1,00 |
| 200_200_00 | 2,05 | 0,72/0,73 | 0,96/0,98 | 1,00/1,00 | 0,29/0,30 | 0,85/0,86 | 1,00/1,00 |
| 200_800_00 (EPR3) | 4,11 | 0,73/0,77 | 1,00/1,00 | 1,00/1,00 | 0,61/0,59 | 0,95/0,94 | 1,00/1,00 |
| 200_800_10 | 3,32 | 0,73/0,75 | 0,99/1,00 | 1,00/1,00 | 0,43/0,50 | 0,92/0,91 | 1,00/1,00 |
| 200_800_20 | 1,05 | 0,61/0,57 | 0,82/0,79 | 0,98/0,97 | 0,26/0,22 | 0,65/0,64 | 0,95/0,95 |

Dla odniesienia: na teście bazowym (stany czyste) sieci K = 6 mają czułość dla klasy termicznej 0,64–0,84 przy N = 100 i 0,96–1,00 przy N = 10³ (`runs_eq/eval_shots_Kscan_K6.json`). Wariant B i K = 36 dają te same wnioski (`wyniki.json`, klucz `klasyf`). Przy N = 100 sieci Ahmeda mylą stan głównie z klasą „num”, a bazowe z klasami „gkp” i „binomial”, i tak samo robią na rekordach symulowanych.

**Testy istotności** (χ² jednorodności rozkładu klas, osobno dla każdej sieci, każdego N i każdego zestawu w zakresie treningowym danej rodziny):

| grupa | exp_los vs sym_gauss | exp_los vs sym_term | exp_blok vs sym_gauss |
|---|---|---|---|
| baza K6 A (99 testów) | p_min 0,033; 2 z p < 0,05 | 0,018; 2 | 0,066; 0 |
| baza K6 B (99) | 0,022; 1 | 0,016; 1 | 0,068; 0 |
| Ahmed K6 A (117) | 0,060; 0 | 0,032; 2 | 0,012; 3 |
| Ahmed K6 B (117) | 0,012; 3 | 0,017; 2 | 0,00027; 4 (1 po Bonferronim) |
| baza K36 B (45) | 0,13; 0 | 0,026; 1 | 0,090; 0 |
| Ahmed K36 B (36) | 0,029; 1 | 0,074; 0 | 0,13; 0 |

Łącznie w zakresie treningowym 513 testów na parę. exp_los vs sym_gauss daje 7 wyników z p < 0,05 przy oczekiwanych 26, a exp_los vs sym_term daje 8. Żaden z nich nie przechodzi poprawki Bonferroniego. Jedyny wynik istotny po poprawce (exp_blok, Ahmed K6 B, zestaw 200_800_20, N = 10⁴, p = 0,0003) opiera się na 13 blokach na sieć i dwóch rekordach w klasie „num” wobec zera. Przy takich liczbach przybliżenie χ² nie obowiązuje. Poza zakresem (n̄ = 0,56) różnice exp − sym sięgają p ≈ 10⁻³–10⁻⁴ przy N ≥ 10³. Sieci są tam rozdarte między „thermal” i „binomial” (baza) albo „thermal”, „num” i „gkp” (Ahmed), więc małe różnice rozkładu przesuwają odpowiedź. Jest to zachowanie poza zakresem treningowym, nie defekt danych.

**Wniosek.** Na rekordach eksperymentalnych w zakresie treningowym sieci zachowują się jak na rekordach symulowanych ze stanu termicznego o zmierzonym n̄, przy każdym N. Klasa „thermal” wygrywa w 96–100% rekordów przy N = 10³ (sieci bazowe) i we wszystkich przy N = 10⁴. Asymetria 0,3–1,15 dB (ściśnięty stan termiczny) nie zmienia odpowiedzi: sym_gauss i sym_term nie różnią się istotnie (p_min 0,02 w 99–117 testach). Podobnie jak dla GKP jest to test modelu szumu zliczeń i zgodności rodziny ze źródłem, a nie certyfikacja stanu.

## 5. Ograniczenia

1. Funkcja modu pochodzi z Asavanant+ 2019 przeskalowanej o 10³, bo praca źródłowa odsyła do niedostępnego suplementu. Odtworzenie −4,5/−4,3 dB wobec −4,7/−4,5 dB wskazuje, że różnica jest drugorzędna.
2. Mierzone są tylko dwie fazy (x i p, w osobnych nagraniach). Rozłożenie na K = 6 lub 36 kątów zakłada niezmienniczość fazową, której stan spełnia tylko w przybliżeniu (Vp/Vx = 0,3–1,15 dB). Sieci nie widzą więc pełnej tomografii, a jedynie dwie prawdziwe fazy i rzuty uśrednione po fazie.
3. Szum elektroniki wchodzi do rekordów jako równoważna strata (η_el = 0,97 przy pełnym wzmocnieniu, 0,41 przy „20 dB”). n̄ rekordów jest więc n̄ stanu po tej stracie.
4. Powolny dryf wariancji (Levene) w 4 z 14 zestawów i pozostała korelacja pakietów (|C| ≤ 0,003 przy odstępie rekordu 40 pakietów) nie wpływają na klasyfikację: exp_blok nie różni się od sym_gauss w zakresie treningowym.
5. Sieci Ahmeda K = 6 wytrenowała równoległa analiza PSS. K = 36 policzyłem tylko na 5 zestawach (wariant B). χ² na rzadkich tablicach jest konserwatywne: p < 0,05 wychodzi rzadziej niż 5%.
6. Stan zredukowany EPR ma ⟨x⟩ = ⟨p⟩ = 0 i jest gaussowski, więc test sprawdza jedynie łatwą klasę (thermal). Mówi tyle, że potok nie wprowadza artefaktów przy prawdziwych danych homodynowych, a nie nic o trudnych klasach niegaussowskich.

## 6. Proponowany tekst do pracy (EN)

We also processed the public raw homodyne traces of the optical EPR source of Kawasaki et al. (Dryad, CC0) into temporal-mode quadratures with the 40 ps wave packet of the original work, calibrated to the shot noise, and reproduced the reported two-mode correlations within 0.3 dB. The quadratures of one mode form records of a nearly thermal Gaussian state with mean photon numbers from 0.6 to 4.4, controlled by the pump power and the detection loss. For the eleven settings inside the trained range, the networks assign these records to the thermal family as often as records simulated from a thermal state with the measured photon number, at every N from $10^2$ to $10^4$, and none of the comparisons is significant after correction for multiple tests. Since only the $x$ and $p$ quadratures were measured, we distributed the events over the $K$ phases assuming phase invariance, which the reduced state satisfies to within 1.2 dB.

## Pliki

- `wspolne.py` (wczytanie HDF5, funkcja modu, kwadratury pakietów), `krok12_kwadratury.py`, `krok3_rekordy.py`, `krok4_klasyf.py`, `krok5_agreguj.py`, `uruchom_krok4.sh` (K = 6 A), `uruchom_krok4b.sh` (pozostałe zestawy)
- Dane: `dane/*.json` (metryki kwadratur pakietów), `rekordy/*.json` (rekordy jednego modu, testy), `klasyf/<grupa>_<wariant>/*.json` (zliczenia klas), `wyniki.json` (zbiorczo). Logi uruchomień i surowe pliki HDF5 nie są częścią repozytorium (patrz `MANIFEST_LARGE_FILES.md`).
- Bez rysunków. Ewentualne figury powinny czytać tylko `wyniki.json` i `rekordy/*.json`.
