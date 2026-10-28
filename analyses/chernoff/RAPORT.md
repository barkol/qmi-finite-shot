# Wykladniki Chernoffa: heterodyna Q, homodyna W (K=36), granica kwantowa

Skrypty: chernoff.py (samotest analityczny: dwa stany koherentne C_Q = C_W = |da|^2/4, QCB = |da|^2, zgodne do 1e-9),
podsumuj.py. Dane: out/M24_*.npz (24 stany na klase, samplery i kanaly jak test sweep, siatka 128 x 128, okno +-12),
tabela out/tabela_M24.txt, out/podsumowanie_M24.json. Kontrola: QCB >= C_Q i QCB >= C_W we wszystkich parach.

## Wnioski
1. Idealny obserwator: xi_W / xi_Q (mediana) 1.12-1.21 czysty, 1.03-1.11 defazowanie, 0.87-1.25 straty (zalezy od zbioru).
   Roznica detekcji to najwyzej ok. 25% w wykladniku, czyli co najwyzej 1.25x wiecej zdarzen. Nigdzie wyraznej przewagi Q.
2. Obie detekcje leza 5-10x ponizej kwantowej granicy Chernoffa (QCB); straty zblizaja je do granicy (xi/xi_QCB ok. 0.2 przy 80%).
3. Po klasach: Q lepsza dla stanow rozniacych sie gruboziarniscie (koherentny 0.47 vs 0.38, sciśnieta proznia), W dla
   struktury drobnej i radialnej (Fock, termiczny, kot, GKP, kody liczbowe). Zgodne z postulatem skali w przestrzeni fazowej.
4. Siec vs idealny obserwator: przewaga Q sieci przy N = 10^2 NIE wynika z wykladnikow; to efekt przetwarzania
   (FBP wzmacnia szum zliczen w obrazie W) albo uczenia. Znak W przy defazowaniu i duzym N zgodny z xi_W > xi_Q,
   ale wielkosc (ok. 10 pkt) duzo wieksza niz przewiduje idealny obserwator (bledy ~0 juz przy 10^3).
## Zastrzezenia
- xi liczone wzgledem najblizszego z 24 probek na klase: zawyza prawdziwe infimum (klasy sa ciagle); stosunek W/Q stabilniejszy.
- P_e(N) ~ 1/2 exp(-N xi) pomija prefaktory; sluzy tylko do znaku i rzedu wielkosci.
- Test rozstrzygajacy punkt 4: pary S-W (sinogram bez FBP), patrz paper/results/numbers.tex (klucze `sw_*`).
