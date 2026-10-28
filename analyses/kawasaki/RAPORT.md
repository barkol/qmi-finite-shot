# Rekordy PSS Kawasaki i in. (Nat. Commun. 2024) w protokole testu GKP

Dane: Dryad doi:10.5061/dryad.9p8cz8wqn (CC0), 11 plików `quad_01GHz_{01,03,10,25}mW_{00..20}dB[_2].mat`,
6 faz (60–210° co 30°), po 5000 zdarzeń na fazę (30 000 na zestaw). Praca: Kawasaki i in., Nat. Commun. 15 (2024),
doi:10.1038/s41467-024-53408-w; preprint arXiv:2402.17408.
Wszystkie liczby pochodzą z `out/summary.json` (generuje go `agg.py`), a tabele z `out/tabele.md`.

## 1. Metoda (adaptacja `qmi/x8/konno.py`)

Kod: `kawa.py` (etapy), `agg.py` (testy i tabele), `fig.py` (rysunek, dane w `out/fig_data.npz`), `ahm_k6.sh` (sieci Ahmeda K=6).
Oryginały w projekcie nie były edytowane. Surowe `.mat` czytam tylko przez `scipy.io.loadmat`, Pythona uruchamiam z `-I`,
a wejście przechodzi przez strażnika (6 faz, 5000 skończonych liczb, |x| < 0,9·XMAX). Każdy etap zapisuje osobny punkt
kontrolny `out/<zestaw>/<etap>.npz` i przy wznowieniu pomija gotowe. Najpierw przeszedł tani test na jednym i na dwóch plikach.
Ustawienia: QMI_XMAX=12, siatka 128, CPU ≤10 procesów, sieci na GPU.

* **Kąty.** Tak jak dla Konno (`qmi.exp_gkp`): fazy 60, 90, 120, 150° bez zmian, a 180° → 0° i 210° → 30° ze zmianą znaku
  (x_{θ+180°} = −x_θ). Dostajemy dokładnie zestaw kątów potoku K=6 {0, 30, …, 150°}.
* **kalib.** Dopasowuję średnią przy fazie a modelem c + x0 cos a + p0 sin a i odejmuję tylko offset c, jak dla Konno.
  ⟨n⟩ liczę z drugich momentów: dla 6 równo rozłożonych kątów w [0,180°) średnia ⟨x_θ²⟩ jest dokładnie równa (⟨x²⟩+⟨p²⟩)/2.
* **skala (nowy etap, bo brak osobnego rekordu próżni).** Skanuję s (x_potok = s·x_dane) dla s od 0,707 do 1,414 i liczę
  logL z jakobianem N log s dla (i) MaxLik i (ii) fizycznego modelu PSS (strata η ≤ 1 po a S(r,ψ)|0⟩).
  s = 1 odpowiada próżni o wariancji 1/2, a s = √2 próżni o wariancji 1/4, jak podaje README.
* **stac, iid.** Kopie `konno.stac` (7 testów kolejności, R=200 permutacji, Bonferroni) i `konno.iid`
  (χ² jednorodności 10 bloków; odległość bloku i losowego podzbioru od histogramu całości).
* **ml.** MaxLik RρR jak w `konno.ml` (D=40, 3000 iteracji, POVM całkowany po pikselach siatki potoku).
  Bootstrap obejmuje 20 replik (1500 iteracji każda).
  **Uwaga metodyczna:** logL liczona na POVM jest identyczna dla obu kierunków obrotu, bo ρ* przy −sgn daje te same rozkłady.
  Kierunek wybieram więc tak jak w konno.py, czyli po zgodności z marginesami potoku (radon obrazu W).
  Zysk logL wynosi od 151 do 1465 na korzyść sgn=+1 we wszystkich 11 zestawach.
* **pss.** Wiarygodność multinomialna modelu PSS z parametrami (r, ψ, η). Liczę wierność Uhlmanna F(model, ρ_ML)
  oraz maksymalną wierność ρ_ML z czystym PSS i z kotem nieparzystym.
* **cent (wariant czułości, spoza protokołu Konno).** Każdą fazę centruję osobno.
* **negN.** MaxLik z 50 losowych podzbiorów N ∈ {100, 10³, 10⁴}, licząc udział podzbiorów z W(0) < 0.
* **klasyf.** Jak `konno.klasyf`: obrazy FBP z 6 kątów, 200 losowych podzbiorów rekordu, wszystkie rozłączne kolejne
  bloki i 200 rekordów symulowanych z marginesów potoku ρ_ML (te same N i fazy). Dla całego rekordu (N = 3·10⁴) biorę
  1 obraz, 200 replik bootstrapu i 200 rekordów symulowanych po 5000 zdarzeń na kąt.
* **Sieci.** Wszystkie mają K=6, po 3 ziarna:
  * bazowe `runs_eq/clf_W_shotsK6_s*` (te same co dla Konno);
  * `runs_eq/clf_W_gkpxK6_s*`;
  * Ahmed: **w `runs_ahm_eq` nie ma sieci K=6** (są tylko K=36, K=12 i sino K=36). Wytrenowałem je więc tym samym
    przepisem co `clf_W_shotsK6` (`qmi.shots_data --copies 2 --K 6` z `data_eq_ahm/ahm/{train,val}/mixed`,
    `qmi.clf train`, 25 epok, ziarna 0–2) do `kawasaki/runs_ahmK6/` (duży plik poza gitem, patrz `MANIFEST_LARGE_FILES.md`). Dokładność sprawdziłem na
    `data_eq_ahm/ahm/test/sweep`: 85,9% przy N=100 bez straty i 99,3% przy N=10⁴. Sieci bazowe K=6 mają odpowiednio 94,3% i 100%.
  * **Sieci Q i S nie mają tu zastosowania.** Sieci Q wymagają heterodyny, a sieci S sinogramu z 36 kątów, a dane mają
    6 faz homodynowych. Dla Konno też użyto wyłącznie W z K=6.
* **Testy różnic exp/sym.** Najpierw powtórzyłem testy z `make_numbers_konno.py` (χ² jednorodności rozkładu klas dla
  każdej sieci i N). **Przy 5000 zdarzeniach na fazę ten test nie jest poprawny.** 200 losowych podzbiorów jednego
  krótkiego rekordu silnie na siebie zachodzi (przy N=10⁴ jest to 1667 z 5000 zdarzeń na kąt), a χ² zakłada niezależne
  losowania. Naiwny χ² daje stąd p do 10⁻⁹, a bootstrap całego rekordu p do 10⁻²².
  Dlatego dodałem dwa testy:
  1. **rektest (test na poziomie rekordu).** Statystyka T to TVD między rozkładem klas 50 podzbiorów jednego rekordu
     a rozkładem z 400 niezależnych rekordów symulowanych. Rozkład zerowy daje 40 pełnych rekordów symulowanych
     (5000 na kąt), z których podzbiory losuję tak samo, bez zwracania (rozkład hipergeometryczny na zliczeniach pikseli).
     Taki test ma tę samą strukturę zależności co dane.
  2. **Kolejne bloki wobec sym.** Używam dokładnego testu permutacyjnego tabeli 2×K zamiast χ². Przy N=10⁴ są tylko
     2 bloki: χ² dawał wtedy p=7·10⁻⁵, a test dokładny p≈0,07.

## 2. Skala kwadratur: dane mają już konwencję projektu (próżnia 1/2)

Zdanie z README „ħ = 1/2” (próżnia 1/4) jest **niespójne z danymi i z pracą**. Praca podaje [x̂,p̂]=i (ħ=1).
Nie przeskalowuję danych (s=1). Za tym przemawiają trzy niezależne kontrole:

1. **Model PSS + strata.** Przy s=√2 (próżnia 1/4 według README) logL modelu jest gorsza o 1642 do 5531 we wszystkich
   11 zestawach, a dopasowanie ląduje na granicy (r=1,5; η≈0,1–0,2). Przy s=0,707 (próżnia 1) logL jest gorsza
   o 835 do 1993 (poza 20 dB, gdzie różnica wynosi 39). Optimum s*_PSS = 0,95–1,0 dla wszystkich zestawów do 15 dB.
   Proste oszacowanie analityczne: przy próżni 1/4 wariancje z 1 mW (0,99–1,33) wymagałyby η ≈ 1,8 > 1.
2. **Zgodność z opublikowaną ujemnością.** W tabeli 2 (s=1) min W wynosi −0,096(4), −0,086(5) i −0,066(6)
   dla 3, 10 i 25 mW, a praca podaje −0,097(5), −0,088(5) i −0,066(6). Dla 1 mW min W = −0,092(3) wobec −0,090(4);
   W(0) = −0,077(3) jest mniejsze co do modułu, bo minimum jest przesunięte (p0 = −0,18). Przy s=√2 W(0) wynosiłoby
   od −0,02 do −0,05, a ⟨n⟩ od 1,6 do 2,8, co jest niefizyczne dla PSS przy słabej pompie.
3. **MaxLik bez modelu** wyklucza tylko s=0,707 (ΔlogL od −270 do −1907). Dla s ≥ 1 wiarygodność jest płaska,
   z zyskiem +44 do +88 przy s=√2 (zbyt elastyczny model przy D=40). Ta kontrola sama nie rozstrzyga, ale nie przeczy s=1.

Sprawdzian kalibracji „na próżni z 20 dB” zaproponowany w zleceniu nie działa. Strata jest dodawana **za wzmacniaczem
fazoczułym** („we add additional loss after the second OPA”), więc stan przy 20 dB nie dąży do próżni:
Var wynosi od 0,91 do 1,18, a ⟨n⟩ = 0,54. Strata pogarsza jedynie efektywną sprawność pomiaru.

## 3. Tabele

### 3.1 Kalibracja, stacjonarność, niezależność (zadanie 1)

| zestaw | c | x0 | p0 | χ²/3 | Var min..max | ⟨n⟩ momenty | stac istotne/7 | iid Fisher p | blok/los N=10⁴ [95%] |
|---|---|---|---|---|---|---|---|---|---|
| 01mW_00dB | +0.104(16) | +0.025 | −0.184 | 40.8 | 0.989..1.329 | 0.686(7) | 0 | 0.324 | 0.92 [0.77, 1.07] |
| 03mW_00dB | +0.015(16) | +0.051 | −0.038 | 4.1 | 0.918..1.512 | 0.759(7) | 0 | 0.305 | 1.20 [1.13, 1.28] |
| 10mW_00dB | +0.037(18) | +0.046 | −0.053 | 15.6 | 0.852..1.958 | 0.933(9) | 0 | 0.853 | 0.82 [0.73, 0.92] |
| 25mW_00dB | −0.031(18) | −0.018 | −0.009 | 2.9 | 0.554..2.483 | 1.112(9) | 0 | 0.670 | 0.98 [0.80, 1.15] |
| 01mW_00dB_2 | +0.012(15) | −0.030 | +0.039 | 28.3 | 0.962..1.280 | 0.632(7) | **3** | **0.006** | 1.39 [1.15, 1.64] |
| 01mW_01dB_2 | +0.021(15) | −0.009 | +0.054 | 93.7 | 0.994..1.256 | 0.655(7) | 0 | 0.600 | 0.92 [0.82, 1.02] |
| 01mW_02dB_2 | +0.143(16) | +0.112 | −0.190 | 19.6 | 1.017..1.429 | 0.750(8) | 0 | 0.017 | 0.96 [0.83, 1.09] |
| 01mW_05dB_2 | −0.021(15) | −0.009 | +0.009 | 2.5 | 0.923..1.281 | 0.617(6) | 0 | 0.472 | 1.04 [0.87, 1.21] |
| 01mW_10dB_2 | −0.033(16) | −0.010 | −0.001 | 14.1 | 0.996..1.370 | 0.714(7) | 0 | 0.034 | 1.07 [1.03, 1.12] |
| 01mW_15dB_2 | −0.011(15) | +0.010 | −0.025 | 16.6 | 0.878..1.264 | 0.597(7) | 0 | 0.279 | 1.02 [0.93, 1.12] |
| 01mW_20dB_2 | +0.038(14) | +0.027 | −0.033 | 8.6 | 0.911..1.181 | 0.538(8) | 0 | 0.287 | 0.90 [0.81, 0.98] |

* Model średnich (offset + przesunięcie) często nie wystarcza (χ² do 94 przy 3 st. swobody). Średnie faz mają
  nieregularne odchyłki o wielkości 0,1–0,19, czyli kilka σ. Wariant z osobnym centrowaniem każdej fazy zmienia
  W(0) o mniej niż 0,012 (tab. 3.3).
* Stacjonarność: 10 z 11 zestawów przechodzi wszystkie 7 testów. W `01mW_00dB_2` istotne są ANOVA K10 i K50 oraz
  Levene K200 (dryf średniej), co potwierdza Fisher p=0,006 i stosunek blok/los 1,39. Rozrzutu fazy nie da się
  oszacować metodą z konno.py, bo |dVar/dθ| < 0,9 przy większości kątów (słabe ściśnięcie). Tam, gdzie się da, wynosi ≤ 0,5°.
* Kurtoza waha się od −0,85 do −1,1 dla ≤ 10 dB, czyli rozkład jest dwumodalny, jak dla |1⟩ (−1). Przy 20 dB
  spada do −0,45, co oznacza zbliżanie się do rozkładu gaussowskiego.
* Przedziały blok/los przy N=10⁴ opierają się tylko na 3 blokach na kąt, więc są kruche.

### 3.2 Rekonstrukcja MaxLik i model PSS (zadanie 2); niepewności z bootstrapu

| zestaw | czystość | ⟨n⟩ | W(0) | min W | min W/W(0) praca | TVD ML max (połowa/połowa) | PSS r [dB] | η | F(model,ML) | F czysty PSS | F kot nieparz. (\|α\|) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01mW_00dB | 0.546(6) | 0.697(8) | −0.077(3) | −0.092(3) | −0.090(4) | 0.047 (0.070) | 0.11 [0.9] | 0.64 | 0.970 | 0.615 | 0.615 (0.45) |
| 03mW_00dB | 0.530(6) | 0.769(9) | −0.096(4) | −0.096(4) | −0.097(5) | 0.031 (0.060) | 0.16 [1.4] | 0.68 | 0.974 | 0.649 | 0.649 (0.56) |
| 10mW_00dB | 0.502(7) | 0.947(10) | −0.083(4) | −0.086(5) | −0.088(5) | 0.036 (0.062) | 0.29 [2.5] | 0.70 | 0.948 | 0.627 | 0.626 (0.74) |
| 25mW_00dB | 0.502(5) | 1.133(10) | −0.066(6) | −0.066(6) | −0.066(6) | 0.057 (0.065) | 0.47 [4.0] | 0.65 | 0.989 | 0.596 | 0.595 (0.98) |
| 01mW_00dB_2 | 0.537(3) | 0.650(7) | −0.080(4) | −0.081(4) | – | 0.035 (0.061) | 0.09 [0.7] | 0.63 | 0.996 | 0.625 | 0.625 (0.42) |
| 01mW_01dB_2 | 0.544(3) | 0.672(6) | −0.092(3) | −0.093(3) | – | 0.039 (0.064) | 0.08 [0.7] | 0.65 | 0.994 | 0.642 | 0.642 (0.39) |
| 01mW_02dB_2 | 0.513(6) | 0.765(8) | −0.067(4) | −0.085(4) | – | 0.035 (0.068) | 0.11 [0.9] | 0.66 | 0.932 | 0.594 | 0.594 (0.47) |
| 01mW_05dB_2 | 0.527(3) | 0.637(4) | −0.072(3) | −0.073(3) | – | 0.042 (0.055) | 0.09 [0.8] | 0.62 | 0.998 | 0.612 | 0.612 (0.42) |
| 01mW_10dB_2 | 0.466(5) | 0.724(7) | −0.048(4) | −0.049(4) | – | 0.032 (0.079) | 0.11 [0.9] | 0.63 | 0.942 | 0.568 | 0.568 (0.47) |
| 01mW_15dB_2 | 0.464(4) | 0.606(7) | +0.008(6) | −0.003(1) | (praca: „negativity remained”) | 0.032 (0.062) | 0.13 [1.1] | 0.53 | 0.961 | 0.483 | 0.483 (0.44) |
| 01mW_20dB_2 | 0.453(3) | 0.546(9) | +0.093(7) | −0.002(1) | (praca: brak ujemności) | 0.027 (0.068) | 0.11 [1.0] | 0.44 | 0.912 | 0.340 | 0.340 (0.45) |

* ρ_ML odtwarza 6 marginesów potoku z TVD ≤ 0,057. To mniej niż TVD między dwiema połowami rekordu (0,055–0,079),
  czyli dopasowanie mieści się w szumie zliczeń. Iteracje są zbieżne (zmiana logL w ostatnich 500 iteracjach < 0,01).
* ⟨n⟩ z MaxLik zgadza się z momentami (np. 0,697 wobec 0,686).
* Model PSS ma 3 parametry. Sprawność efektywna η wynosi od 0,62 do 0,70 do 10 dB i spada do 0,53 przy 15 dB
  i 0,44 przy 20 dB. Ściśnięcie rośnie z pompą: 0,9, 1,4, 2,5 i 4,0 dB. Wierność modelu z ρ_ML mieści się
  w przedziale 0,91–0,998, a TVD modelu sięga 0,07–0,10 przy 1 mW, bo model nie zawiera przesunięcia ani szumu nadmiarowego.
* Wierność z kotem nieparzystym jest umiarkowana (0,6–0,65 do 10 dB, |α| od 0,4 do 1,0). To mały kot, a przy słabej
  pompie praktycznie stan |1⟩ ze stratą (p1 ≈ 0,61, p0 ≈ 0,36).
* Przy 15 dB różnię się od pracy: W(0) = +0,008(6), a min W = −0,003(1), więc ujemność jest na granicy (3σ, przy dodatnim
  obciążeniu MaxLik). Praca pisze, że przy ok. 15 dB ujemność pozostaje. Przy 20 dB wynik zgadza się z pracą (brak ujemności).

### 3.3 Ujemność przy skończonym N (zadanie 5) i wariant centrowania

| zestaw | udział W(0)<0, N=100 | N=10³ | N=10⁴ | mediana W(0) N=10³ | kurtoza śr. | centrowanie per faza: W(0) / min W |
|---|---|---|---|---|---|---|
| 01mW_00dB | 0.76 | 1.00 | 1.00 | −0.067 | −0.92 | −0.071 / −0.087 |
| 03mW_00dB | 0.92 | 1.00 | 1.00 | −0.098 | −0.99 | −0.095 / −0.096 |
| 10mW_00dB | 0.90 | 1.00 | 1.00 | −0.087 | −1.00 | −0.084 / −0.084 |
| 25mW_00dB | 0.74 | 1.00 | 1.00 | −0.064 | −0.99 | −0.065 / −0.065 |
| 01mW_00dB_2 | 0.78 | 1.00 | 1.00 | −0.074 | −0.94 | −0.073 / −0.079 |
| 01mW_01dB_2 | 0.80 | 1.00 | 1.00 | −0.084 | −0.94 | −0.083 / −0.087 |
| 01mW_02dB_2 | 0.68 | 1.00 | 1.00 | −0.060 | −0.93 | −0.079 / −0.083 |
| 01mW_05dB_2 | 0.54 | 1.00 | 1.00 | −0.059 | −0.93 | −0.072 / −0.073 |
| 01mW_10dB_2 | 0.54 | 0.98 | 1.00 | −0.047 | −0.86 | −0.049 / −0.049 |
| 01mW_15dB_2 | 0.14 | 0.32 | 0.16 | +0.008 | −0.71 | +0.011 / −0.003 |
| 01mW_20dB_2 | 0.08 | 0.00 | 0.00 | +0.089 | −0.45 | +0.095 / −0.003 |

Podzbiory N=10⁴ (1667 z 5000 zdarzeń na kąt) silnie się pokrywają, więc ich udziały nie są niezależnymi próbami.

### 3.4 Klasyfikacja (zadanie 3) i kontrola modelu szumu zliczeń (zadanie 4)

Wartości w komórce: klasa dominująca i jej udział (3 sieci razem) w losowych podzbiorach exp i w rekordach sym;
p_rek to test na poziomie rekordu; p_blok to dokładny test permutacyjny kolejnych bloków (minimum z 3 sieci).
Dla całego rekordu: klasy z 3 sieci.

**Sieci bazowe (shotsK6):**

| zestaw | N=100 | N=10³ | N=10⁴ | cały rekord |
|---|---|---|---|---|
| 01mW_00dB | fock .63/.60; p_rek .05; p_blok .65 | fock .89/.89; .56; .67 | fock .67/.70; .41; 1 | fock,fock,cat |
| 03mW_00dB | fock .65/.69; .98; .12 | fock .88/.90; .12; .66 | fock .81/.76; .71; .55 | fock,fock,cat |
| 10mW_00dB | fock .41 / cat .41; .78; .38 | cat .58/.60; .05; .75 | cat .73/.81; .34; .51 | fock,cat,cat |
| 25mW_00dB | cat .74/.81; .41; .49 | cat 1.00/1.00; 1; 1 | cat 1.00/1.00; 1; 1 | cat,cat,cat |
| 01mW_00dB_2 | fock .71/.66; .95; .51 | fock .97/.98; .22; .08 | fock .79/.81; 1; 1 | fock,fock,cat |
| 01mW_01dB_2 | fock .73/.73; .46; .69 | fock .97/.99; .76; .03 | fock .68/.75; .44; .63 | fock,fock,cat |
| 01mW_02dB_2 | fock .69/.62; .32; .13 | fock .94/.93; .80; .05 | fock .67/.73; .29; 1 | fock,fock,cat |
| 01mW_05dB_2 | fock .68/.66; .61; .28 | fock .98/.99; .80; .02 | fock .92/.90; .95; 1 | fock,fock,cat |
| 01mW_10dB_2 | fock .62/.64; .80; .02 | fock .96/.96; .78; .11 | fock .98/.98; 1; 1 | fock,fock,fock |
| 01mW_15dB_2 | fock .41/.40; .73; .25 | fock .82/.79; .54; .61 | fock .96/.96; 1; 1 | fock,fock,fock |
| 01mW_20dB_2 | binomial .35/.34; .71; .42 | binomial .60/.61; .83; .62 | binomial .51/.52; .49; .05 | binomial,fock,fock |

Pełne tabele dla trzech zbiorów sieci (z udziałem rekordów sym w klasie całego rekordu) są w `out/tabele.md`.

**Sieci gkpx K6:** obraz jest ten sam. Fock dla 1–10 mW i dla całej serii strat do 20 dB (przy 20 dB i N=100 przeważa
thermal), kot dla 25 mW (0,99–1,00 przy N ≥ 10³).

**Sieci Ahmeda K6 (wytrenowane w tym audycie):** Fock dla 1–10 mW i dla strat do 20 dB (przy 15–20 dB i N=100
przeważa coherent). **Dla 25 mW wynik to binomial, nie kot** (0,82 przy N=10³, 1,00 przy N=10⁴, cały rekord 3/3),
choć zbiór Ahmeda zawiera koty z |α| ∈ [2, 3]. Koty Ahmeda mają jednak ⟨n⟩ ≈ 4–9 i nie przypominają małego kota
z ⟨n⟩ ≈ 1,1.

**Zbiorcze testy exp wobec sym** (11 zestawów × 3 N; dla rektest p z B=40, więc najmniejsze możliwe p to 0,024):

| sieci | rektest: istotne <0,05 / 33 (min p) | po sieciach / 99 | bloki (dokładny): <0,05 / 99 (min p; Bonferroni) | cały rekord: min udział sym w klasie exp |
|---|---|---|---|---|
| baza | 2 (0,049) | 2 | 3 (0,017; 1) | 0,415 |
| gkpx | 0 (0,098) | 1 | 2 (0,016; 1) | 0,33 |
| ahm | 4 (0,024) | 5 | 6 (0,0038; 0,38) | 0,765 |

Przy prawdziwej H0 spodziewamy się ok. 1,7 z 33 i 5 z 99 wyników poniżej 0,05. Żadna różnica nie przeżywa poprawki
na wielokrotne testy. Najmniejsze p bloków (sieci Ahmeda, N=10³) przypada na `01mW_00dB_2` i `01mW_01dB_2`.
Pierwszy z tych zestawów jest tym, w którym testy stacjonarności wykryły dryf.
Naiwny χ² (protokół Konno) dawałby tu „istotne” różnice (min p 10⁻⁹), ale jest to artefakt zależności podzbiorów
(punkt 1). Dla Konno problem jest słabszy, bo rekord był 4 razy dłuższy, ale przy N=10⁴ warto to sprawdzić (zob. ograniczenia).

## 4. Seria strat 0–20 dB (zadanie 5)

* **Ujemność W.** ρ_ML ma W(0) od −0,067 do −0,092 do 2 dB, około −0,07 przy 5 dB, −0,048(4) przy 10 dB,
  +0,008(6) przy 15 dB i +0,093(7) przy 20 dB. Kurtoza maleje co do modułu od −0,94 do −0,45, a η modelu PSS
  spada z 0,63 do 0,44. Przejście od stanu niegaussowskiego z ujemnością do stanu bez ujemności następuje między 10 a 15 dB.
* **Przy skończonym N** W(0) < 0 występuje w 54–92% podzbiorów N=100 do 10 dB (8–14% przy 15–20 dB), a przy N=10³
  w 98–100% do 10 dB, 32% przy 15 dB i 0% przy 20 dB.
* **Klasyfikacja (sieci bazowe).** Do 15 dB dominuje klasa Fock. Przy N=10⁴ i dla całego rekordu udział Focka rośnie
  w zakresie 5–15 dB (z 0,67–0,8 do 0,96–0,98), bo zanika domieszka klasy cat, którą jedna z trzech sieci wybiera
  dla całego rekordu przy ≤ 5 dB. Przy 20 dB klasa zmienia się na binomial dla N ≤ 10⁴ (udział Focka 0,16–0,47),
  a cały rekord dostaje binomial, fock, fock. Żadna sieć nie wybiera klasy gaussowskiej (thermal, coherent, vacuum)
  dla N ≥ 10³ nawet przy 20 dB, choć W jest tam już nieujemne. Niegaussowski kształt rozkładu (kurtoza −0,45) pozostaje
  i jest wyraźny, co zgadza się z uwagą w pracy („non-Gaussian quadrature feature remains”).
* **Pompa.** Przy 0 dB sieci bazowe zmieniają decyzję z Fock (1 i 3 mW) na cat (10 i 25 mW). Odpowiada to ⟨n⟩ od 0,70
  do 1,13 i ściśnięciu od 0,9 do 4 dB, czyli przejściu PSS od |1⟩ do małego kota.
* Rekordy symulowane z ρ_ML odtwarzają te przejścia w każdym zestawie (rys. f, kółka puste przy pełnych markerach).

Rysunek: `fig_kawasaki.pdf`, `fig_kawasaki.png` i `fig_kawasaki_gray.png`. Panele (a)–(d) pokazują W_ML; w wersji
szarej znak niosą kontury W=0 (linia ciągła) i W=−0,05 (przerywana). Panel (e) pokazuje W(0) i min W wobec straty,
(f) udział klasy Fock (sieci bazowe; pełne markery to exp, puste to sym), a (g) udział W_ML(0) < 0 przy różnych N.

## 5. Wnioski

1. Dane są już w konwencji projektu (próżnia 1/2), a README myli się co do ħ. Rekonstrukcja MaxLik w protokole potoku
   odtwarza opublikowane ujemności do ~0,002.
2. Rekordy przechodzą testy niezależności i stacjonarności z jednym wyjątkiem (`01mW_00dB_2`, dryf średniej).
3. Odpowiedź sieci na rekordach eksperymentalnych nie różni się od odpowiedzi na rekordach symulowanych z ρ_ML.
   Pokazują to test na poziomie rekordu, dokładny test bloków i klasa całego rekordu, we wszystkich trzech zbiorach sieci,
   bez istotnych wyników po poprawce. Model szumu zliczeń potoku opisuje więc także te dane. Ponieważ ρ_ML pochodzi
   z tych samych danych, test dotyczy modelu szumu, nie stanu.
4. Nazwa stanu zależy od rodzin treningowych. Sieci bazowe i gkpx nazywają PSS przy słabej pompie stanem Focka
   (fizycznie poprawnie: ~|1⟩ ze stratą), a przy 10–25 mW kotem. Sieci Ahmeda nazywają stan 25 mW stanem binomialnym.
   Stany PSS mają ⟨n⟩ = 0,55–1,13, czyli **poza zakresem treningowym** zbioru bazowego (⟨n⟩ ∈ {2,…,5}), a tym bardziej
   zbioru Ahmeda.
5. Seria strat pokazuje zanik ujemności W między 10 a 15 dB. Decyzja sieci zmienia się dopiero przy 20 dB i dla N ≥ 10³ nie
   przechodzi do rodzin gaussowskich, co jest zgodne z utrzymującym się niegaussowskim kształtem rozkładu.

## 6. Ograniczenia

* Nie ma rekordu próżni, więc skalę ustalam pośrednio (model PSS, zgodność z W z pracy). Offsety faz nie są stałe
  (χ² dopasowania średnich do 94/3). Protokół odejmuje wspólny offset, a wariant z osobnym centrowaniem faz zmienia W(0) ≤ 0,012.
* Zakładam, że kolejność zdarzeń w pliku odpowiada kolejności pomiaru (jak dla Konno). README tego nie mówi.
* Mamy po jednym rekordzie na ustawienie i tylko 5000 zdarzeń na fazę. Przy N=10⁴ są tylko 2–3 rozłączne bloki
  (słaba moc testu bloków), a rektest ma B=40, czyli rozdzielczość p ≈ 0,024.
* Wszystkie sieci mają K=6. Sieci Ahmeda K=6 nie pochodzą z pracy: wytrenowano je na potrzeby tej analizy (3 ziarna, przepis
  `clf_W_shotsK6`, 25 epok) i nie mają ewaluacji drift ani hold-out. Sieci Q i S nie dają się zastosować do tych danych.
* Stany PSS leżą poza rozkładem treningowym (małe ⟨n⟩), więc klasy należy czytać jako „najbliższą rodzinę”, nie jako identyfikację.
* MaxLik z D=40 i 6 fazami ma obciążenie w stronę stanów fizycznych. Przy 15–20 dB min W ≈ −0,003 może być resztą
  tego obciążenia. Bootstrap ma tylko 20 replik.
* Kontrola dla Konno: tam podzbiory przy N=10⁴ (1667 z ~20 000 zdarzeń na kąt) zachodzą na siebie słabiej, ale naiwny χ²
  wobec 200 losowań nadal nie jest ściśle poprawny. Dlatego dla Konno liczymy rektest (`konno_rek.py`, wynik w
  `out_konno/konno_rek.json`) zamiast naiwnego χ².
* Pomocnicze dane treningowe `cache_ahmK6/` (2,2 GB) można usunąć, bo da się je odtworzyć z `ahm_k6.sh`.

## 7. Propozycja tekstu do pracy (styl podrozdziału o GKP)

> We apply the same protocol to the public homodyne records of photon-subtracted squeezed states~\citep{Kawasaki2024}, with $5000$ events at each of six phases for four pump powers and for losses from $0$ to $20$~dB added after the phase-sensitive amplifier.
> The reconstructed states reproduce the published Wigner negativities ($-0.096$, $-0.086$ and $-0.066$ against $-0.097$, $-0.088$ and $-0.066$ at $3$, $10$ and $25$~mW).
> Because random subsets of such short records overlap, we draw the subsets of simulated records in the same way from full simulated records, and none of the $99$ comparisons per network set, nor any comparison of consecutive blocks, is significant after correction for multiple tests.
> The base-set networks assign the states at $1$ and $3$~mW to the Fock family and those at $10$ and $25$~mW to the cat family, although their $\langle\hat n\rangle$ of $0.6$ to $1.1$ lies below the training range, whereas networks trained on the set of Ref.~\cite{Ahmed2021} call the $25$~mW state binomial.
> As the added loss grows, the negativity of $W$ vanishes between $10$ and $15$~dB, but at $N\ge10^3$ no network switches to a Gaussian family.

(Uwaga: sieci K=6 dla zbioru Ahmeda nie należą do planu 135 sieci pracy.)

## 8. Pliki

* `kawa.py`, `agg.py`, `fig.py`, `ahm_k6.sh`: kod.
* `out/<zestaw>/{kalib,stac,iid,ml,skala,pss,cent,negN}.npz`, `klasyf_{baza,gkpx,ahm}.json`, `rektest_{baza,gkpx,ahm}.npz`: punkty kontrolne.
* `out/summary.json`: wszystkie liczby. `out/tabele.md`: tabele. `out/eval_shots_ahmK6.json`: dokładność sieci Ahmeda K6.
* `runs_ahmK6/clf_W_shotsK6_s{0,1,2}`: sieci. `logs/`: logi.
* `fig_kawasaki.{pdf,png}`, `fig_kawasaki_gray.png`: rysunek.
