# MaxLik zamiast FBP dla homodyny: analiza klasyfikacji

Pytanie: czy przewaga heterodyny (Q) nad homodyną z FBP (W) przy skończonej liczbie zdarzeń N znika, gdy obraz Wignera
liczymy z rekonstrukcji największej wiarygodności (W_ML), a nie z filtrowanej projekcji wstecznej. Odniesienia to sieć
heterodynowa Q i sieć S, która czyta surowe histogramy homodynowe.

## 1. Metoda

**Zdarzenia (bitowo te same co w pracy).** `gen_events.py` odtwarza generator z `qmi.eval_shots.gen` (SEED=424242, dla każdego
stanu najpierw heterodyna, potem homodyna) i zapisuje całkowite zliczenia K=36 kątów × 128 binów pikseli. Strażnik:
FBP(zliczenia) == obraz W z oryginalnej funkcji, różnica 0.0 (bitowo) we wszystkich komórkach. Te same stany, te same
zdarzenia i ta sama kolejność co w `runs_eq/pary`, więc W_ML jest sparowane z Q i z W(FBP) na poziomie zdarzeń.
Drugi zestaw (`sw`) odtwarza `qmi.eq.pary_sw.records` (SEED=7121). To dokładnie te zdarzenia, które czyta sieć S
w `pary_sw`, stąd S−W_ML też jest sparowane zdarzenie w zdarzenie (strażnik ≤1.2e-7).

**MaxLik** (`mlcore.py`, adaptacja `qmi.x8.konno.ml`; oryginał nie był ruszany). Element POVM to
Π_{θ,j} = ∫_{bin j} |x_θ⟩⟨x_θ| dx (podsiatka 8 punktów na piksel), ⟨x_θ|n⟩ = e^{-inθ} ψ_n(x), ħ=1 (konwencja `qt.wigner`).
Iteracje RρR startują z ρ=I/D i idą wsadowo na GPU (torch complex64, 700 stanów naraz). Zliczenia normalizuję do sumy 1,
a p_k ograniczam od dołu przez 1e-12, co chroni fp32. Stany z NaN liczyłem ponownie w complex128, ale nie wystąpił żaden.
- Konwencja jest sprawdzona na stanach z obrazem dokładnym. TVD między tr(Πρ) a znormalizowanym `radon(W_obraz)` wynosi
  ≤0.03 (największa dla sqvac i GKP). Znak obrotu sgn=+1 (dla −1 TVD dochodzi do 0.99). Baza Wignera W(|b⟩⟨a|) odtwarza
  obraz danych z dokładnością 2e-8.
- Wymiar Focka D=40. Ogon populacji poza D=30 sięga 2% (sqvac, GKP), poza D=40 najwyżej 0.7%. Wariant D=20 (osobno
  rekonstruowany i trenowany) daje te same wnioski (§3.3).
- Zbieżność: przy N=100 ΔlogL<1e-3 po 100 iteracjach, przy N=1000 3e-4 po 300. Trafności nie zmieniają się od 300 do
  20000 iteracji. W produkcji: 1000 iteracji, 2.55 ms/iterację na 700 stanów (L40S), czyli 2.6 s na komórkę.
- Granica N→∞ (dokładne rzuty, loss 0%): po 100 iteracjach clf_W_exact na W_ML daje 99.5%, więc obraz W_ML jest w tej samej
  konwencji co dane. Przy dalszych iteracjach trafność spada do 91.8%: MaxLik dopasowuje wtedy artefakty interpolacji
  w `radon` obrazu pikselowego (zob. ograniczenia).

**Obraz W_ML** liczę jako W = Re Σ ρ_ab W(|a⟩⟨b|) na siatce 128×128, ±12.

**Klasyfikatory W_ML.**
(a) `clf_W_exact_s0-2` (obrazy dokładne).
(b) `clf_W_ml_s0-2`: ta sama architektura i procedura (`qmi.clf train`, 25 epok, AdamW 3e-4, bs 64), trenowane na obrazach
W_ML z rekordów treningowych. Rekordy treningowe mają **te same zdarzenia co trening sieci shots**: `gen_train_events.py`
odtwarza `qmi.shots_data` (copies=2, seed=31337, N losowe z {1e2,…,1e6,0}), strażnik FBP == shots_data daje 0.0. Dla N=0
jako granicę MaxLik biorę dokładny obraz W. Liczba rekordów: 28 700 treningowych i 5 950 walidacyjnych.
Kontrolnie: (c) `clf_W_shots_s0-4` (sieci FBP) czytające W_ML.

**Statystyka.** `paired_diff` skopiowane z `make_numbers.py`: składnik stanów warstwowo po rodzinach plus składnik ziaren
(Welch), oba w kwadraturze. Strażnik w `analyze.py`: trafności Q i W oraz różnice Q−W i S−W (te same zdarzenia)
odtworzone z plików `pary*` zgadzają się z `ci.json` i `ci_ahm.json` co do 1e-9.

## 2. Nakład obliczeń (łącznie ok. 1 h 30 min, GPU + CPU)
- Zdarzenia testowe (2 tryby × 8 komórek): 2.2 min CPU. Zdarzenia treningowe: około 1 min na zbiór (14 procesów).
- MaxLik: test 42 s, trening i walidacja bazowego 5.5 min, Ahmeda 10 min (GPU dzielone z treningiem).
- Trening 3 ziaren: 13 min (3 równolegle na GPU1), dla Ahmeda i D=20 po około 20 min (6 równolegle).

## 3. Wyniki (punkty procentowe; CI 95%)

### 3.1 Zbiór bazowy (data_eq/qmi/test/sweep, 700 stanów na komórkę; Q, W, S po 5 ziaren, W_ML 3 ziarna)

| komórka | N | Q | W(FBP) | S | **W_ML** (b) | W_ML (a) exact | Q−W | **Q−W_ML** | **S−W_ML** | **W_ML−W** |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 100 | 95.9 | 91.0 | 95.8 | **95.9** | 25.0 | +4.9 [+2.1,+7.7] | **+0.1 [−2.3,+2.4]** | −0.7 [−2.3,+1.0] | **+4.8 [+2.4,+7.3]** |
| loss50 | 100 | 81.3 | 74.8 | 82.3 | 74.3 | 18.2 | +6.5 [+3.0,+10.0] | **+7.0 [+1.5,+12.5]** | +6.4 [+3.6,+9.1] | −0.5 [−5.6,+4.6] |
| loss80 | 100 | 58.3 | 48.5 | 55.3 | 46.9 | 16.8 | +9.8 [+6.1,+13.6] | **+11.5 [+7.9,+15.0]** | +6.6 [+3.2,+9.9] | −1.6 [−4.7,+1.5] |
| deph20 | 100 | 82.7 | 68.6 | 81.5 | 75.2 | 15.8 | +14.1 [+9.4,+18.7] | **+7.4 [+3.5,+11.4]** | +6.2 [+2.8,+9.5] | +6.7 [+2.3,+11.0] |
| clean | 1000 | 99.9 | 99.9 | 100.0 | 100.0 | 51.6 | +0.0 [−0.2,+0.2] | −0.1 [−0.2,+0.1] | −0.0 [−0.1,+0.1] | +0.1 [−0.1,+0.2] |
| loss50 | 1000 | 98.5 | 98.0 | 99.6 | 99.0 | 32.4 | +0.5 [−1.0,+2.0] | −0.5 [−1.5,+0.4] | +0.5 [−0.2,+1.1] | +1.0 [−0.4,+2.5] |
| loss80 | 1000 | 84.6 | 81.1 | 87.8 | 84.8 | 28.0 | +3.5 [−0.3,+7.3] | −0.1 [−3.8,+3.6] | +3.9 [+1.4,+6.4] | +3.6 [+0.2,+7.1] |
| deph20 | 1000 | 98.4 | 98.6 | 99.6 | 99.4 | 17.3 | −0.2 [−1.4,+1.0] | −1.0 [−2.3,+0.2] | −0.2 [−0.7,+0.3] | +0.8 [+0.0,+1.6] |

S w tabeli to sieci S na zdarzeniach `pary_sw`, a S−W_ML liczę na tych samych zdarzeniach. Kolumna „S” różni się więc
o ułamki punktu od `rule_*_S` w pracy, które pochodzą z osobnego losowania. W_ML na zdarzeniach `sw` różni się od W_ML na
zdarzeniach `ev` o ≤1.1 pkt (`wyniki_*.json`, klucz `ML_sw`).
Sieci FBP czytające W_ML (c) dają 85.1 / 62.1 / 41.5 / 56.7 przy N=100, czyli mniej niż na własnym FBP.

Na poziomie rodzin przy N=100 (Q/W/W_ML) zysk MaxLik w clean przychodzi z fock (98/91/94), thermal (79/66/80) i binomial
(96/91/99). Przy stratach 50–80% W_ML nie pomaga GKP, binomial ani fock, np. loss80 GKP 54/27/27, binomial 37/20/19.

### 3.2 Zbiór Ahmeda (data_eq_ahm/ahm/test/sweep; Q i W po 3 ziarna, S 5, W_ML 3, `clf_W_ml` trenowane na rekordach Ahmeda)

| komórka | N | Q | W | S | W_ML | Q−W | Q−W_ML | S−W_ML | W_ML−W |
|---|---|---|---|---|---|---|---|---|---|
| clean | 100 | 94.1 | 84.6 | 88.7 | 90.3 | +9.5 [+4.8,+14.2] | +3.8 [−0.5,+8.1] | −3.2 [−5.8,−0.6] | +5.7 [+1.1,+10.2] |
| loss50 | 100 | 69.5 | 63.4 | 65.1 | 63.6 | +6.1 [−2.0,+14.2] | +5.9 [+0.9,+10.9] | +5.4 [+0.9,+10.0] | +0.2 [−7.2,+7.5] |
| loss80 | 100 | 40.0 | 37.6 | 44.1 | 39.8 | +2.3 [−4.4,+9.1] | +0.1 [−6.2,+6.5] | +4.5 [−3.3,+12.3] | +2.2 [−3.5,+7.9] |
| deph20 | 100 | 83.7 | 72.7 | 82.7 | 78.0 | +11.0 [+4.4,+17.6] | +5.7 [−0.7,+12.0] | +5.7 [−1.3,+12.7] | +5.3 [−1.1,+11.8] |
| clean | 1000 | 99.2 | 98.6 | 99.2 | 99.2 | +0.6 [−0.8,+1.9] | +0.0 [−2.3,+2.3] | −0.3 [−1.6,+1.0] | +0.6 [−1.7,+2.8] |
| loss50 | 1000 | 95.4 | 92.3 | 94.5 | 94.5 | +3.1 [−0.8,+7.0] | +0.9 [−2.2,+4.0] | −0.2 [−2.7,+2.4] | +2.2 [−1.5,+5.9] |
| loss80 | 1000 | 68.7 | 69.0 | 73.4 | 72.9 | −0.4 [−6.3,+5.6] | −4.2 [−10.7,+2.3] | −0.8 [−7.0,+5.5] | +3.8 [−2.5,+10.1] |
| deph20 | 1000 | 97.1 | 95.4 | 98.4 | 97.3 | +1.7 [−1.8,+5.2] | −0.1 [−3.2,+2.9] | +1.1 [−0.1,+2.3] | +1.9 [−0.4,+4.1] |

### 3.3 Odporność na D
Przy D=20 (osobna rekonstrukcja testu i treningu oraz 3 nowe ziarna) wyniki praktycznie się nie zmieniają, np. N=100:
clean Q−W_ML +0.8 [−1.7,+3.3], loss50 +6.1, loss80 +10.8, deph20 +7.4 (`wyniki_base_D20.json`).

### 3.4 Dlaczego sieci „exact” zawodzą na W_ML
MaxLik z 100 zdarzeń zawyża energię. ⟨n⟩_ML wynosi 5.5 wobec prawdziwego 3.5 (clean), 3.9 wobec 1.75 (loss50) i 2.4 wobec
0.7 (loss80). Mediana wierności wynosi 0.46–0.73. Przy N=1000 ⟨n⟩ jest bliskie prawdy (+0.3), a wierność 0.78–0.94
(`diag_base.json`; dla Ahmeda podobnie). Obrazy W_ML są fizyczne, ale leżą poza rozkładem obrazów dokładnych. Trafność
sieci exact (17–58%) mierzy więc przesunięcie rozkładu, a nie informację w W_ML. Dlatego wynikiem głównym jest wariant (b).

## 4. Odpowiedzi
1. **N=100: MaxLik nie zamyka różnicy Q−W ogólnie, w przeciwieństwie do sieci S.** Zamyka ją tylko dla stanów czystych
   (bazowy: z +4.9 do +0.1 [−2.3,+2.4]; Ahmed: z +9.5 do +3.8 [−0.5,+8.1], tam W_ML jest nawet lepsze od S o 3.2 pkt).
   Przy defazowaniu 20% zmniejsza ją mniej więcej o połowę (+14.1 → +7.4 [+3.5,+11.4]). Przy stratach 50% i 80% nie daje
   nic ponad FBP (W_ML−W = −0.5 i −1.6). Q−W_ML wynosi tam +7.0 i +11.5, a S−W_ML pozostaje istotnie dodatnie (+6.4, +6.6,
   w deph20 +6.2). Sieć czytająca surowe histogramy wyciąga więc z tych samych zdarzeń więcej niż MaxLik plus CNN.
2. **N=1000: W_ML nie wygrywa z Q.** W bazowym zbiorze Q−W_ML mieści się w [−1.0, +0.1] pkt i wszystkie CI zawierają 0.
   U Ahmeda najmocniejszy kierunek „W_ML > Q” to loss80: −4.2 [−10.7,+2.3], nieistotne. W_ML jest wtedy nie gorsze od
   W(FBP), a w loss80 lepsze o +3.6 [+0.2,+7.1]. Sugestia, że przy N=1e3 rekonstrukcja MaxLik daje homodynie przewagę
   nad heterodyną (Fernandes i in. 2025, w kontekście tomografii), nie przenosi się na klasyfikację w tej konfiguracji.
   Wynik to remis, a sieć S jest w loss80 lepsza od W_ML o +3.9 [+1.4,+6.4].
3. Reguła decyzyjna z pracy (heterodyna przy małym N i stratach) pozostaje w mocy także wobec rekonstrukcji MaxLik.
   Deficyt homodyny przy N=1e2 dotyczy potoku „rekonstrukcja, potem obraz” (FBP i MaxLik), a nie samych rekordów
   homodynowych, bo sieć S go nie ma.

## 5. Ograniczenia
- **Niedopasowanie modelu.** Zdarzenia potoku pochodzą z `radon` obrazu pikselowego z interpolacją liniową, a MaxLik
  zakłada idealny POVM kwadraturowy. TVD ≤0.03 jest pomijalne przy N≤1e3 (szum próbkowania jest dużo większy), ale przy
  N→∞ MaxLik dopasowuje artefakty (99.5% → 91.8% w teście granicy). Rekordy treningowe z N=1e5 i 1e6 rekonstruowałem
  1000 iteracjami, co działa jak wczesne zatrzymanie. Dla N=0 w treningu użyłem obrazu dokładnego zamiast MaxLik.
- MaxLik bez regularyzacji (bez kary energii, bez ograniczenia rzędu). Przy N=100 zawyża ⟨n⟩. Mniejsze D=20 nie zmienia
  wyników klasyfikacji. Wariantów bayesowskich ani z karą energii nie sprawdzałem.
- W_ML ma 3 ziarna (Q, W, S po 5 w zbiorze bazowym), co poszerza składnik ziaren CI. U Ahmeda Q i W mają po 3 ziarna.
  Przy loss80 N=100 trafności są bliskie szansy (40%), więc CI są szerokie.
- Sieci (b) trenowano tylko na N losowanym jak w treningu shots, bez strojenia hiperparametrów. Najlepsza strata walidacyjna
  (0.165–0.18, trafność wal. 93–94%) jest porównywalna z siecią FBP (0.194, 93%).
- Tylko komórki reguły i N=1e2 oraz 1e3. N=1e4 pominąłem, bo tam Q−W ≈ 0 już z FBP.

## 6. Propozycja do pracy (EN)
> Replacing filtered back-projection by maximum-likelihood reconstruction (iterative RρR, Fock cutoff 40, classifier
> retrained on the reconstructed Wigner functions) removes the heterodyne advantage at $N=10^2$ only for pure states
> ($\Delta_{Q-W_{\rm ML}}=0.1\,[-2.3,2.4]$ points); it halves the gap under 20\% dephasing and leaves it unchanged under
> 50--80\% loss, where a network reading the raw homodyne histograms still closes it. At $N=10^3$, $W_{\rm ML}$ and $Q$
> are statistically indistinguishable in all four rule cells, so the decision rule is unaffected: the small-$N$ deficit
> of homodyne detection belongs to the reconstruct-then-classify pipeline, not to the homodyne records themselves.

## 7. Pliki (`analyses/maxlik/`)
- kod: `mlcore.py`, `gen_events.py`, `gen_train_events.py`, `run_ml.py`, `classify.py`, `analyze.py`, `diag.py`, `conv.py`, `t_konw.py`
- wyniki: `wyniki_maxlik.json` (wszystko razem), `wyniki_base.json`, `wyniki_ahm.json`, `wyniki_base_D20.json`, `diag_*.json`
- dane pośrednie: `ev_*` (zliczenia), `ml_*` (ρ_ML), `cls_*` (poprawność [sieć, stan]), `runs*/clf_W_ml_s*` (sieci (b)), logi `*.log`
