# Czy reguła W/Q zależy od architektury sieci?

**Wniosek:** nie zależy. ResNet-18, trenowany tym samym protokołem co CNN v2, daje we wszystkich 12 komórkach reguły
ten sam liść co CNN v2. Przy N=100 Q−W > 0 z 95% CI powyżej zera we wszystkich czterech warunkach szumu, a od N=1000
nie ma istotnych różnic. Przy N=100 również S≈Q. ResNet-18 jest przy tym nieco *słabszy* od CNN v2 (o 2–8 pp przy N=100),
więc wynik nie bierze się z niedotrenowanej sieci v2, której mocniejsza architektura miałaby zamknąć lukę Q−W.

## 1. Architektura i dlaczego właśnie ta

| | CNN v2 (praca, `qsdl/cnn/classifier.py`) | ResNet-18 „r18” (`analyses/arch2/clf_arch2.py`) |
|---|---|---|
| głębokość | 8 splotów 3×3 (4 bloki Conv-BN-ReLU×2 + MaxPool) | 17 splotów + FC, 8 bloków rezydualnych [2,2,2,2] |
| kanały | 32-64-128-256 | 64-128-256-512 |
| głowica | AdaptiveAvgPool(4) → FC 4096→256 → Dropout 0.5 → FC | **globalny** average pooling → jedna FC 512→7, bez dropoutu |
| połączenia rezydualne | brak | tak |
| pole recepcyjne / downsampling | ×16 (128→8) | stem 3×3 stride 2 bez max-poola, potem ×8 (128→64→…→8); pole recepcyjne obejmuje cały obraz |
| parametry | 2 223 271 | 11 171 271 (5×) |

Uzasadnienie wyboru: to standardowa architektura o innej zasadzie działania (rezydua, głębsza, globalny pooling bez
cech zależnych od położenia, 5× więcej parametrów). Jest więc istotnie inna, a zarazem zwykle mocniejsza, co jest
właściwym sprawdzianem pytania „czy lepsza sieć zamknie lukę Q−W przy małym N”. Stem dopasowany do 1 kanału i 128×128
(conv 3×3/2 zamiast 7×7/2 + max-pool). Ta sama sieć bez zmian obsługuje sinogramy 36×128 (S).

## 2. Protokół (różnica tylko w architekturze)

- Dane: te same rekordy co sieci `runs_eq/clf_{W,Q}_shots_s*` i `clf_W_sino_s*`. Odtworzone na maszynie z GPU tym samym kodem
  i ziarnami (`qmi.shots_data --copies 2`, ziarno 31337, K=36; `qmi.sino_data`, ziarno 5150) ze źródeł
  `data_eq/qmi/{train,val}/mixed`, QMI_XMAX=12. Kontrola: ten sam kod dał na niej pliki **bitowo identyczne** z pamięcią
  podręczną stacji roboczej (pamięć podręczna rekordów shotsK6 w RAM, 2 pliki, wszystkie klucze). Zbiory: 574 pliki treningowe,
  119 walidacyjnych (bez kanału gauss), ~28,7 tys. rekordów treningowych na reprezentację.
- Trening skopiowany 1:1 z `qmi.clf.train`: skala = std zbioru treningowego, AdamW lr 3e-4, batch 64, 25 epok
  (EPOCHS z `qmi/eq/kolejka_eq.py`), bez augmentacji (v2 też ich nie ma), model = minimum straty walidacyjnej (best.pt).
  Punkt kontrolny co epokę + wznowienie (sprawdzone w tanim teście: 2 epoki, wznowienie do 3).
- Sieci: r18 Q, W, S po 3 ziarna (0, 1, 2), GPU0 L40S, trzy tory równolegle.
- Ewaluacja sparowana (`analyses/arch2/pary_arch2.py`): rekordy z `qmi.eval_shots.gen` (K=36, SEED=424242), komórki
  clean = straty 0, loss50, loss80, deph20, N=1e2, 1e3, 1e4, 700 stanów na komórkę. W tej samej macierzy są sieci v2
  (Q, W, S, po 5 ziaren). Strażnik: poprawność v2 stan po stanie zgadza się z `runs_eq/pary` i `runs_eq/pary_sino`,
  z wyjątkiem 14 przypadków (na 15×12×700 = 126 tys. predykcji). We wszystkich 14 margines top-2 logitów jest < 0,017,
  czyli to remis rozstrzygany zaokrągleniem splotu: L40S liczy inaczej niż RTX 5060 Ti. Diagnoza
  `tools/diag_v2_remis.py`: rekordy są identyczne, CPU float64 zgadza się z L40S, a GPU stacji roboczej daje inny wynik.
  Do macierzy wchodzą wtedy wiersze v2 z plików pracy. Lista jest w `pary/rozbieznosci_v2.json`.
- CI: `paired_diff`, `Acc`, `ci_of` wyjęte dosłownie (ast) z `paper/scripts/make_numbers.py`. Strażnik: v2 (5 ziaren)
  odtwarza `paper/results/ci.json` (rule_N*_*_{Q,W,S,dQW,dQS}, 60 kluczy) z dokładnością do 1e-9. Liść wg
  `fig_rule_flow.py`, tylko zbiór bazowy: Q, gdy CI(Q−W) > 0; W, gdy < 0; „none”, gdy CI zawiera zero.

## 3. Koszt i przebieg treningu

- Rekordy 3 min, trening 9 sieci 40 min, pary 1,5 min (jedno GPU).
- Czas epoki r18 przy trzech torach na jednym L40S: obrazy 27–38 s (11–16 min na sieć), sinogramy ~16 s (~6,5 min).
- Walidacja (najlepsza epoka): r18 vl 0,18–0,25, va 0,91–0,95; v2 (s0–2) vl 0,12–0,19, va 0,93–0,96.
  r18 przeucza się wcześniej: najlepsza epoka 1–5 (raz 13), trafność treningowa 0,98–0,99, a dla v2 najlepsza epoka 6–14.
  Strata walidacyjna skacze między epokami w obu architekturach (v2 Q też), stąd wybór po minimum straty.

## 4. Tabela komórek (trafność w %, różnice w pp, 95% CI); v2 = 5 ziaren jak w pracy

| N | komórka | r18 Q / W / S | v2 Q / W / S | r18 Q−W | v2 Q−W (ci.json) | liść r18 / v2 | r18 Q−S | v2 Q−S |
|---|---|---|---|---|---|---|---|---|
| 100 | clean | 94.1 / 88.2 / 92.6 | 95.9 / 91.0 / 96.0 | +5.9 [+1.9, +9.9] | +4.9 [+2.1, +7.7] | **Q** / **Q** | +1.6 [−3.7, +6.8] | −0.0 [−2.4, +2.4] |
| 100 | loss50 | 77.5 / 68.9 / 73.3 | 81.3 / 74.8 / 81.3 | +8.6 [+0.4, +16.8] | +6.5 [+3.0, +10.0] | **Q** / **Q** | +4.1 [−4.0, +12.3] | +0.0 [−3.2, +3.2] |
| 100 | loss80 | 54.9 / 42.0 / 52.0 | 58.3 / 48.5 / 56.0 | +12.9 [+6.1, +19.7] | +9.8 [+6.1, +13.6] | **Q** / **Q** | +2.9 [−4.8, +10.5] | +2.3 [−1.9, +6.6] |
| 100 | deph20 | 79.3 / 65.7 / 78.4 | 82.7 / 68.6 / 82.8 | +13.6 [+8.8, +18.4] | +14.1 [+9.4, +18.7] | **Q** / **Q** | +0.9 [−6.7, +8.5] | −0.2 [−4.1, +3.8] |
| 1000 | clean | 100.0 / 99.8 / 99.7 | 99.9 / 99.9 / 100.0 | +0.2 [−0.4, +0.8] | +0.0 [−0.2, +0.2] | none / none | +0.3 [−0.5, +1.0] | −0.1 [−0.2, +0.1] |
| 1000 | loss50 | 98.6 / 97.0 / 97.2 | 98.5 / 98.0 / 99.4 | +1.5 [−0.3, +3.4] | +0.5 [−1.0, +2.0] | none / none | +1.4 [−0.5, +3.3] | −0.9 [−1.8, +0.1] |
| 1000 | loss80 | 84.0 / 77.1 / 78.3 | 84.6 / 81.1 / 88.8 | +6.9 [−2.8, +16.6] | +3.5 [−0.3, +7.3] | none / none | +5.7 [−1.8, +13.2] | **−4.2 [−7.3, −1.0]** |
| 1000 | deph20 | 96.9 / 96.5 / 99.3 | 98.4 / 98.6 / 99.8 | +0.3 [−3.8, +4.5] | −0.2 [−1.4, +1.0] | none / none | −2.5 [−7.3, +2.3] | **−1.4 [−2.6, −0.2]** |
| 10000 | clean | 100.0 / 100.0 / 99.7 | 100.0 / 100.0 / 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | none / none | +0.3 [−0.5, +1.0] | +0.0 [+0.0, +0.0] |
| 10000 | loss50 | 99.9 / 100.0 / 99.2 | 100.0 / 99.9 / 100.0 | −0.1 [−0.5, +0.3] | +0.1 [−0.2, +0.3] | none / none | +0.7 [−1.2, +2.6] | −0.0 [−0.1, +0.1] |
| 10000 | loss80 | 93.9 / 89.8 / 87.9 | 95.1 / 95.4 / 97.9 | +4.1 [−8.2, +16.5] | −0.3 [−3.9, +3.2] | none / none | +6.0 [−12.3, +24.4] | −2.8 [−6.8, +1.2] |
| 10000 | deph20 | 99.5 / 100.0 / 99.6 | 99.9 / 100.0 / 100.0 | −0.5 [−1.9, +1.0] | −0.1 [−0.2, +0.1] | none / none | −0.0 [−1.3, +1.2] | −0.1 [−0.2, +0.1] |

Różnica architektur w tej samej reprezentacji (r18 3 ziarna − v2 5 ziaren, sparowane po stanach) oraz v2 ograniczone
do ziaren 0–2 (ta sama liczba ziaren co r18):

| N | komórka | r18−v2 Q | r18−v2 W | r18−v2 S | v2 (s0–2) Q−W | liść v2 (s0–2) |
|---|---|---|---|---|---|---|
| 100 | clean | −1.8 [−4.1, +0.5] | −2.8 [−6.3, +0.8] | −3.4 [−8.4, +1.6] | +5.6 [+1.7, +9.4] | Q |
| 100 | loss50 | −3.8 [−12.3, +4.6] | **−5.9 [−10.3, −1.5]** | **−8.0 [−14.1, −1.9]** | +6.6 [+2.1, +11.0] | Q |
| 100 | loss80 | −3.4 [−10.4, +3.6] | **−6.5 [−11.9, −1.1]** | **−3.9 [−7.3, −0.5]** | +8.7 [+3.7, +13.6] | Q |
| 100 | deph20 | **−3.3 [−6.3, −0.3]** | −2.9 [−7.4, +1.7] | −4.4 [−11.2, +2.4] | +14.8 [+9.1, +20.5] | Q |
| 1000 | clean | +0.1 [−0.1, +0.2] | −0.1 [−0.7, +0.4] | −0.3 [−1.0, +0.5] | +0.0 [−0.4, +0.4] | none |
| 1000 | loss50 | +0.1 [−1.7, +1.8] | −1.0 [−2.6, +0.7] | **−2.2 [−3.6, −0.7]** | +0.8 [−1.0, +2.6] | none |
| 1000 | loss80 | −0.6 [−7.8, +6.6] | −4.0 [−14.0, +6.0] | **−10.5 [−16.8, −4.2]** | +2.8 [−4.4, +10.0] | none |
| 1000 | deph20 | −1.5 [−5.9, +2.8] | −2.1 [−5.8, +1.6] | −0.5 [−1.1, +0.1] | +0.0 [−2.5, +2.6] | none |
| 10000 | clean | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | −0.3 [−1.0, +0.5] | +0.0 [+0.0, +0.0] | none |
| 10000 | loss50 | −0.1 [−0.5, +0.3] | +0.1 [−0.2, +0.3] | −0.8 [−2.8, +1.2] | +0.1 [−0.5, +0.7] | none |
| 10000 | loss80 | −1.2 [−12.2, +9.9] | −5.7 [−18.4, +7.1] | −10.0 [−30.6, +10.5] | −1.5 [−7.8, +4.8] | none |
| 10000 | deph20 | −0.4 [−1.8, +1.0] | +0.0 [+0.0, +0.0] | −0.4 [−1.4, +0.5] | +0.0 [+0.0, +0.0] | none |

(Tabele generuje `analiza.py` z `wyniki.json`, plik `tabela.md`.)

## 5. Wnioski

1. **Reguła Q−W się nie zmienia.** Liść r18 = liść v2 w 12/12 komórkach. Przy N=100 Q−W jest dodatnie i istotne we
   wszystkich czterech warunkach, w wielkości zgodnej z v2 (+5,9 do +13,6 pp wobec +4,9 do +14,1 pp). Od N=1000 brak
   istotnych różnic. Sieć v2 ograniczona do 3 ziaren daje te same liście, więc zgodność nie wynika z liczby ziaren.
2. **Znak:** przy N=100 r18 ma Q−W dodatnie we wszystkich komórkach, a punktowo nawet większe niż v2 przy stratach
   (loss50 +8,6, loss80 +12,9). Przy N≥1000 znaki średnich są mieszane i małe, wszystkie CI obejmują zero.
3. **S≈Q przy N=100 potwierdzone** (r18 Q−S od +0,9 do +4,1 pp, wszystkie CI obejmują zero). Przy N=1000 jest jedna
   różnica. v2 ma S istotnie lepsze od Q w loss80 (−4,2 pp) i deph20 (−1,4 pp), a r18 tego nie pokazuje: Q−S jest
   nieistotne, przy loss80 nawet ze znakiem dodatnim. Powód: sinogramowy r18 jest najsłabszy względem v2
   (S: −10,5 pp przy N=1000 loss80). Teza „S co najmniej tak dobre jak Q” przy N≥1000 jest więc cechą v2, nie obu
   architektur. Teza „S≈Q przy N=100” jest od architektury niezależna.
4. **Trafności spadają, nie rosną.** r18 jest gorszy od v2 we wszystkich komórkach N=100: Q o 1,8–3,8 pp, W o 2,8–6,5 pp,
   S o 3,4–8,0 pp. Istotnie w 5 z 12 porównań. Przy N≥1000 oba są przy suficie, z wyjątkiem loss80, gdzie r18 ma duży
   rozrzut ziaren (np. Q przy N=1e4: 88/96/97%). Przyczyna: 5× więcej parametrów, ten sam protokół bez augmentacji
   i bez harmonogramu lr. r18 przeucza się po 1–5 epokach. Mocniejsza nominalnie sieć nie zmniejsza luki Q−W,
   W traci względem v2 nawet więcej niż Q.
5. Zastrzeżenia: tylko zbiór bazowy (eqn), bez zbioru Ahmeda, więc liście „Q/W w jednym zbiorze” z `fig_rule_flow.py`
   nie są tu sprawdzane. 3 ziarna r18 wobec 5 v2. Hiperparametry nie były dostrajane pod r18. To celowe, bo różnić
   ma się tylko architektura, ale nie mierzymy przez to najlepszej możliwej trafności ResNetu.

## 6. Propozycja do pracy (EN)

> To test whether the rule depends on the classifier, we retrained all finite-record networks with a ResNet-18
> (residual blocks, global average pooling, 11.2M parameters, five times more than our CNN) under the identical
> protocol and evaluated it on the same paired test records. The decision leaf is unchanged in all twelve cells:
> at $N=10^2$ heterodyne beats homodyne by 6 to 14 points (95% CI above zero in every noise condition), and from
> $N=10^3$ on no difference is significant. The larger network is slightly less accurate, not more, so the advantage of
> $Q$ at small $N$ reflects the measurement record rather than limited classifier capacity.

## 7. Pliki

- `clf_arch2.py`: model r18 + trening (protokół qmi.clf). `pary_arch2.py`: ewaluacja sparowana ze strażnikiem.
  `analiza.py`: CI i liście (`wyniki.json`) oraz render (`tabela.md`). `run_cloud.sh`: cały potok na maszynie z GPU (wznawialny).
- `runs/r18_{Q_shots,W_shots,W_sino}_s{0,1,2}/{best.pt,history.json}` (bez ckpt.pt).
- `pary/*.npz` (y, names, correct; 24 sieci × 700 stanów), `pary/rozbieznosci_v2.json` (14 remisów v2).
- `logs/` (dane, test, trening, pary). `det/`: próbka bitowej zgodności rekordów. `tools/diag_v2_remis.py`.
- Nie przechowywane: `data` (rekordy, 2,4 GB) i `ckpt.pt`.
