# Atrybucje GradientSHAP sieci Q, W, S i odniesienie do idealnego obserwatora

Katalog: `analyses/shap/`. Python (conda env `ml`), `QMI_XMAX=12`, captum 0.9.0, GPU.

| plik | rola |
|---|---|
| `shap_compute.py` | obliczenia (wznawialne, jeden npz na komórkę w `out/`): `conv`, `inputs`, `attr`, `hell`, `sanity` |
| `shap_summary.py` | metryki -> `out/summary.json`, mapy klasowe wyrównane -> `out/maps.npz` |
| `shap_extra.py` | odniesienia zerowe dla (c) i test ilorazu wiarygodności -> `out/extra.json` |
| `shap_plot.py` | figury (tylko wczytuje `maps.npz`, `summary.json`, `extra.json`) |
| `shap.log`, `run_full*.out` | logi (zero błędów); `out_small/` = tani test 4 stany/klasę |

## Metoda
- Stany: zbiór testowy `data_eq/qmi/test/sweep`, 40 stanów na klasę (pierwsze 40 z pliku, ⟨n⟩ cyklicznie 2..5), 7 klas.
  Komórki: czyste (`loss`, poziom 0) i defazowanie 20% (`dephase`, poziom 20).
- Wejścia jak w `qmi.eval_shots`: dokładne (Q; FBP z K=36 dokładnych rzutów; sinogram N=0) oraz R=5 realizacji
  rekordów N=100 (`heterodyne_image`, `homodyne_wigner`, `sinogram_counts`, K=36). Sieć `clf_W_exact` dostaje dokładny W
  (na nim była trenowana; na FBP miała tylko 39% trafień).
- Sieci: `clf_{Q,W}_shots_s0,s1`, `clf_W_sino_s0,s1`, dodatkowo `clf_{Q,W}_exact_s0` na wejściach dokładnych.
  Trafność przy N=100, czyste, s0: Q 0.86-1.00, W 0.69-1.00, S 0.81-1.00 (zgodna z eval_shots).
- GradientSHAP względem logitu prawdziwej klasy, 64 próbki, `stdevs=0`. Baseline główny: 20 losowych wejść tego samego typu
  z innych klas (dla rekordów: rekordy N=100). Kontrola: obraz próżni i zero. Dodatkowo wariant bez mnożenia przez (x-b)
  (`multiply_by_inputs=False`, oczekiwany gradient, dalej EG) jako mapa czułości.
- Każda mapa normowana do sum|a|=1 przed uśrednianiem (każdy stan i realizacja z tą samą wagą).
- Orientacja: faza jest w meta dla `coherent`, `cat` (`phase`) i `sqvac` (`psi`, oś ściskania). Obraz obracany o fazę,
  sinogram przesuwany cyklicznie po osi kątów (krok 5°) z odbiciem osi położenia przy przejściu przez 180°. Znak ustalony
  empirycznie: korelacja wyrównanych Q stanów koherentnych 0.86 (zły znak 0.12), sinogramów 0.86 (0.22); sqvac 0.75 -> 0.98.
  `fock`, `thermal` są obrotowo symetryczne. `gkp` i `binomial` nie mają losowej fazy w samplerze (`qmi/eqn_states.py`),
  więc zwykłe uśrednianie jest poprawne i profil radialny nie był potrzebny.
- Idealny obserwator: dla stanu ρ najbliższy (max współczynnik Bhattacharyya) σ z innych klas wśród tych samych 280 stanów;
  mapa Hellingera H_Q = |√p - √q|² na siatce heterodyny (p = dokładne Q, s = 1/2). Analogicznie H_S w przestrzeni sinogramu
  (rozkłady brzegowe K=36 kątów, C = -(1/K) Σ log BC_k). Mediany C_1/2: czyste het 0.084, hom 0.104; defazowanie 0.021, 0.025.
- Sanity (Adebayo i in. 2018): kaskada od góry (fc2, fc1, blok 4..1; Conv/Linear/BN reset, statystyki BN zerowane),
  70 stanów czystych (10/klasę), s0, wejścia rekord i dokładne. Do każdej metryki (b), (c) liczona jest ta sama wartość dla
  sieci całkowicie przelosowanej: to jest odniesienie zerowe, bez którego liczby (a)-(c) są nieinterpretowalne.

## Najważniejsza obserwacja metodyczna (sanity)
Mediana Spearmana mapy sieci wytrenowanej z mapą po randomizacji (rekordy N=100; kolejno fc2 ... całość):

| | po fc2 | całość losowa |
|---|---|---|
| SHAP ze znakiem, Q / W / S | -0.18 / 0.06 / 0.13 | 0.02 / 0.00 / 0.01 |
| \|SHAP\|, Q / W / S | 0.98 / 0.80 / 0.97 | **0.97 / 0.62 / 0.97** |
| \|EG\|, Q / W / S | 0.53 / 0.50 / 0.66 | -0.25 / 0.05 / -0.25 |

- Mapy ze znakiem i mapy EG przechodzą test. **Mapy \|SHAP\| sieci Q i S na rekordach go nie przechodzą**: przy 100 zliczeniach
  czynnik (x-b) jest niezerowy tylko w pikselach ze zliczeniami, więc wielkość atrybucji wyznacza położenie zdarzeń, nie sieć.
  Na wejściach dokładnych jest podobnie (Q 0.94, S 0.87). Sieć W (gęste wejście FBP) przechodzi częściowo (0.62).
- Wniosek praktyczny: liczby oparte na \|SHAP\| dla Q i S trzeba czytać jako "gdzie są dane", a o sieci mówi dopiero różnica
  względem sieci losowej albo mapa ze znakiem.

## Odpowiedzi
**(a) Struktury fizyczne.** Miara: udział \|SHAP\| w obszarze W<0 (−1% max\|W\|: prążki kota, pierścienie Focka, sieć GKP,
kody dwumianowe) podzielony przez udział powierzchni tego obszaru (wzbogacenie).
- Sieć W, N=100, czyste: Fock 4.7, kot 7.7, GKP 4.4, dwumianowy 6.1 (dokładne: 14.7, 20.6, 6.7, 11.3).
  Podzbiór sanity: wytrenowana 5.6 wobec losowej 2.1 (rekordy), 13.7 wobec 8.2 (dokładne). **Sieć W patrzy na obszary
  ujemnego W ok. 2.6x bardziej, niż wynika z samego wejścia.**
- Sieć Q: wzbogacenie 12.3 / 13.7 / 3.8 / 10.2, ale losowa daje 8.5 wobec 11.7 wytrenowanej (rekordy) i 8.0 wobec 8.6
  (dokładne): efekt prawie w całości wynika z wejścia.
- Kot: na mapach ze znakiem (fig. a) dodatnia atrybucja leży w środku między płatami (obszar prążków) dla Q, W i S
  (w sinogramie: przewężenie przy kątach prostopadłych do osi kota). Ściśnięta próżnia: sieci ważą rdzeń wzdłuż osi
  ściśniętej, nie ogony; udział ogonów (r>2) w atrybucji na nośniku 0.46 (Q), 0.41 (W) wobec 0.80/0.73 powierzchni.
  Fock przy N=100: atrybucja W rozproszona po pierścieniach i szumie, Q skupiona w środku (dołek Q w zerze).
- Stabilność: Spearman map klasowych \|SHAP\| s0 vs s1 (rekordy) Q 0.83-0.98, W 0.82-0.98, S 0.90-0.95; EG 0.52-0.97.

**(b) Rozlanie atrybucji W na szum FBP.** Mediana udziału \|SHAP\| poza nośnikiem stanu (Q>1% max; \|W\|>1% max; \|S\|>1% max),
N=100, czyste: **W 0.46** (rodziny 0.22-0.81, najgorzej ściśnięta próżnia), **Q 0.18**, **S 0.12**; powierzchnia poza nośnikiem
0.93 / 0.87 / 0.69. Sieć losowa: W 0.85, Q 0.12, S 0.10. Wejścia dokładne: W 0.25, Q 0.30, S 0.26. Defazowanie 20%:
W 0.40, Q 0.11, S 0.05.
- Tak: przy N=100 prawie połowa atrybucji sieci W leży na smugach FBP poza stanem; przy wejściach dokładnych już nie (0.25,
  jak Q i S), więc to efekt rekonstrukcji, nie architektury. Sieć W i tak skupia się dwa razy silniej niż sieć losowa (0.46
  wobec 0.85). Niskie wartości Q i S nie świadczą o "lepszym patrzeniu": wynikają z tego, że zliczenia padają tylko na nośnik.
- Mapy czułości (EG): Q 0.96, S 0.91, W 0.62 poza nośnikiem. Sieci Q i S są najbardziej czułe na zdarzenia w pustych
  obszarach (zdarzenie tam przeczy klasie), sieć W na nośniku.

**(c) Zgodność z idealnym obserwatorem.** Spearman map klasowych w obszarze nośnika:
- \|SHAP_Q\| vs H_Q (N=100, czyste): Fock 0.43, koherentny 0.86, sq. próżnia 0.29, termiczny 0.33, kot 0.56, GKP 0.57,
  dwumianowy 0.70 (mediana 0.56; cała siatka 0.57-0.78; defazowanie mediana 0.63). Ale sam stan Q vs H_Q daje medianę 0.58,
  a sieć całkowicie losowa 0.47 wobec 0.48 wytrenowanej (ten sam podzbiór). **Zgodność Q z idealnym obserwatorem jest
  w tej metryce nieodróżnialna od zgodności samych danych.**
- \|SHAP_S\| vs H_S: mediana 0.43 (sq. próżnia 0.01); losowa 0.28 wobec 0.36 wytrenowanej: słaby efekt.
- Sieć W vs H_Q: wytrenowana 0.46 wobec losowej 0.27 (\|SHAP\|), 0.33 wobec 0.08 (EG): jedyny przypadek, gdzie sieć wyraźnie
  podąża za mapą informacji idealnego obserwatora ponad to, co daje wejście.
- EG sieci Q jest antyskorelowany z H_Q (−0.34; losowa +0.48). Sprawdzona hipoteza "sieć Q = detektor ilorazu wiarygodności"
  (czułość ~ \|log p/q\|): 0.37 wobec 0.59 dla sieci losowej, więc **nie potwierdzona**.

## Ograniczenia
- GradientSHAP na rzadkich rekordach zliczeń jest zdominowany przez czynnik (x-b); wnioski o sieciach Q i S opierają się
  na mapach ze znakiem i na różnicach względem sieci losowej, a te liczone są na podzbiorze 10 stanów na klasę.
- Wybór baseline: Spearman map klasowych \|SHAP\| "inne klasy" vs próżnia: Q 0.82-0.93, W 0.93-0.99, S 0.89-0.95 (rekordy);
  baseline zerowy zmienia mapy Q bardziej (koherentny 0.52, kot 0.66). Na wejściach dokładnych sieć W jest wrażliwa na
  baseline (koherentny 0.51, termiczny 0.44 vs próżnia). Wnioski (a)-(c) nie zależą od tego wyboru jakościowo, ale wartości tak.
- Maska nośnika 1% max i maska W<0 są arbitralne; maski z rzeczywistego W, więc dla sieci Q obszar W<0 jest tylko
  geometrycznym odniesieniem.
- H_Q dla najbliższego σ z 40 próbek na klasę (zawyża odległość, jak w audycie Chernoffa); mapa Hellingera to gęstość wkładu
  do C_1/2, nie dokładnie do wykładnika przy optymalnym s.
- Obrót o fazę z interpolacją liniową; sinogram z rozdzielczością 5°.
- Tylko s0 i s1, tylko N=100 i nieskończone N, dwie komórki szumu.

## Wnioski
1. Mapy ze znakiem są specyficzne dla sieci (sanity zdany) i stabilne między seedami; pokazują sensowne struktury: środek kota,
   rdzeń ściśniętej próżni, węzły GKP dla W.
2. Sieć W jest jedyną, dla której ilościowo widać uczenie się struktur nieklasycznych ponad wejście (wzbogacenie W<0 2.6x
   względem sieci losowej, zgodność z H_Q 0.46 wobec 0.27).
3. Przy N=100 sieć W ma ok. 46% atrybucji poza stanem, na szumie FBP (0.25 przy danych dokładnych). To spójne z wnioskiem 4
   analizy Chernoffa: przewaga sieci Q przy małym N wynika z przetwarzania FBP, nie z fizyki pomiaru.
4. Dla Q i S atrybucje na rekordach nie niosą informacji o sieci w samej wielkości (sanity niezdany); ich zgodność z idealnym
   obserwatorem (mediana 0.56) jest równa zgodności samych danych (0.58) i sieci losowej (0.47). Nie należy tego przedstawiać
   jako dowodu, że sieć Q "uczy się optymalnego obserwatora".

## Propozycja do pracy (EN)
Gradient-SHAP maps of the Wigner-image network weight the negative regions of W (cat fringes, Fock rings, the GKP lattice)
about 2.6 times more than a weight-randomized control, but at N=100 about half of the attribution lies outside the state,
on FBP streaks, against a quarter for exact images. For heterodyne and sinogram records the attribution magnitude is fixed
by where the counts fall and fails the randomization test, so it cannot be read as evidence that these networks learn the
ideal-observer weighting.

## Figury
- `fig_shap_maps.pdf/.png/_gray.png` (7.0 x 2.55 in): (a) średnie wyrównane mapy SHAP ze znakiem (N=100, czyste, s0) dla
  Q, W, S i mapa H_Q; rozbieżna niebiesko-pomarańczowa wokół zera, kontury ±30% max (ciągły +, przerywany −), kropkowany
  kontur stanu 15% max; (b) udział \|SHAP\| poza nośnikiem, linie przerywane = sieć losowa; (c) Spearman z H_Q/H_S
  i odniesienie "sam stan".
- `fig_shap_maps_deph20.*` (defazowanie 20%), `fig_shap_maps_EG.*` (oczekiwany gradient, panele b/c dla EG).
- `fig_shap_sanity.pdf/.png/_gray.png` (3.4 x 2.5 in): kaskada randomizacji (znak / \|SHAP\| / \|EG\|) i przykład mapy kota dla
  sieci W.
- Uruchomienie od zera: `python analyses/shap/shap_compute.py --stage all --M 40 --R 5 --nsamp 64` (~35 min GPU), potem
  `shap_summary.py`, `shap_extra.py`, `shap_plot.py`.
