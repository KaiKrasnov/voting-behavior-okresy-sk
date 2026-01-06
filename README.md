## Part 2 — Results model (predikcia podielu hlasov vo volbach)

### Ciel analyzy
V tejto casti sa zameriavame na analyzu faktorov, ktore ovplyvnuju podiel
hlasov v parlamentnych volbach na urovni okresov Slovenskej republiky.
Aj ked hlavnou ulohou je pochopenie vztahov medzi parametrami okresov a
hlasovanim v tych okresoch, vysledny model by este mal fungovat dobre sam osebe
(cize mal by dobre predikovat hlasy pre politicke subjekty na zaklade
parametrov okresu).

---

### Premenne

**Zavisle premenne**
- podiely hlasov pre politicke strany a koalicie.

**Vysvetľujúce premenné**
- demografia: podiel a pocet muzov a zien, podiel a pocet obyvatelov, ktore
  patria roznym vekovym skupinam (do 15 rokov, 15-64 rokov, 65 a viac rokov),
  podiel a pocet obyvatelov so slovenskym obcianstvom, obyvatelov narodenych na
  Slovesnku, obyvatelov-Slovakov a cudzincov,
- urbanizacia: podiel a pocet zijucich v meste a na vidieku,
- vzdelanie: podiel a pocet ludi s roznym najvyssim dosiahnutym vzdelanim,
- trh práce: nezamestnanost, rozne pracovne stavy (pracujuci, student vysokej
  skoly, nezamestnany atd), rozne pracovne vztahy (zamestnanec, podnikatel),
- ekonomika: priemerná mzda.

---

### Metody

Na predikciu podielov hlasov vo volbach bola pouzita **multinomicka logisticka
regresia**, konkretne `sklearn.linear_model.LogisticRegression`. Jej vyhodou je
to, ze ona funguje ako pravdepodobnostny model, cize na rozdiel od linearnej
regresie nepotrebujeme zvlast aplikovat softmax na vysledok - to nebude fungovat
dobre, pretoze linearna regresia (LASSO, RIDGE) bude sa optimalizovat podla
svojich predpisov a nase ohranicenie na to, aby sa vsetko skladalo do 1, bude
ju iba miast.

Aby nasa uloha patrila do mnoziny uloh, ktore vie riesit logisticka regresia,
vieme ju preformulovat na _"ake su pravdepodobnosti hlasovania za jednotlive
politicke strany za predpokladu parametrov okresu volica?"_

Pre hodnotenie, ci dobre alebo zle model predikuje podiely, bol pouzity
**log loss**, pretoze je to zakladna chybova funkcia pre logisticku regresiu
a ona lepsie zohladnuje logaritmicky background metody nez MSE alebo MAE.

Pre ferove rozhodovanie dolezitosti parametra pre model vstupne data su
standartizovane pomocou `sklearn.preprocessing.StandardScaler`. Vplyv parametra
na podiel hlasov za politicku stranu bol definovany ako priemer absolutnych
hodnot koeficientov logistickej regresie zodpovedajucich tomuto parametru.

---

### Vysledky

**Kvalita modelu**

Multinomicka logisticka regresia sa prejavila velmi dobre pre tuto ulohu - pre
testovy okres Bratislava IV model, ktory nebol trenovany na tom okrese, mal
log-loss 2.247, pricom idealny log-loss je 2.243. Hladanie parametrov, ktore
maju najvacsi vplyv na model, tiez nevypada zle, pretoze model, ktory nebol
trenovany na testovom okrese, ale pritom bol trenovany iba na 10 najdenych
dolezitych parametroch, dava log-loss 2.269.

**Najdolezitejsie atributy pre model**

10 atributov okresov, ktore maju najvacsi vplyv na podiel hlasov pre model, su nasledovne:

| Atribut                                                | Priemerny vplyv | Najpositivnejsi vplyv | Najnegativnejsi vplyv |
|:-------------------------------------------------------|:----------------|:----------------------|:----------------------|
| Podiel Slovakov                                        | 0.06            | 0.13 (KDH)            | -0.387 (MKÖ-MKS)      |
| Podiel študentov vysokych škol                         | 0.031           | 0.18 (KDH)            | -0.17 (MKÖ-MKS)       |
| Podiel obyvatelov s iným pracovným stavom              | 0.028           | 0.178 (MKÖ-MKS)       | -0.061 (KDH)          |
| Podiel obyvatelov s iba zakladnym vzdelanim            | 0.027           | 0.09 (MKÖ-MKS)        | -0.126 (SMER)         |
| Podiel obyvatelov s nezistenou štátnou príslušnosťou   | 0.024           | 0.054 (SME RODINA)    | -0.183 (MKÖ-MKS)      |
| Podiel obyvatelov v produktívnom veku                  | 0.024           | 0.143 (MKÖ-MKS)	     | -0.048 (KDH)          |
| Podiel osob v domácnosti                               | 0.022           | 0.134 (MKÖ-MKS)	     | -0.034 (PS-SPOLU)     |
| Podiel obyvatelov s iba strednym vzdelanim s maturitou | 0.021           | 0.044 (VLASŤ)	       | -0.131 (MKÖ-MKS)      |
| Pocet obyvatelov s iným pracovným stavom               | 0.021           | 0.106 (MOST-HÍD)      | -0.052 (ĽSNS)         |
| Podiel obyvatelov s nezisteným pracovným stavom        | 0.02            | 0.136 (MKÖ-MKS)	     | -0.048 (SNS)          |

Aj ked vplyv niektorych parametroch na hlasovanie je zrejmy (napriklad to, ze
zmena koncentracie Slovakov ma vplyv na podporu stran s narodnostnou politikou),
pre vacsinu dolezitych pre model atributov sa neda povedat, aky presne je vztah
medzi tymto parametrom a hlasovanim za jednotlive politicke subjekty:
- Pre niektore vztahy sa da sice vysvetlit logiku, ale to vyzera pritiahnute za
  vlasy. Napriklad pre ten vysledok, ze zvysenie poctu obyvatelov s nezistenou
  statnou prislusnostou zvysuje pravdepodobnost hlasu za konzervativnu a
  anti-imigracnu stranu SME RODINA. To sice moze suvisiet, ale ten koeficient
  je velmi maly a pravdepodobne nie je dolezitym a samostatnym faktorom pre
  hlasovanie za tuto stranu;
- Niektore vztahy vyzeraru vybrate nahodne. Napriklad to, ze zvysenie podielu
  obyvatelov s nanajvys zakladnym vzdelanim pravdepodobne najviac zmensi podiel
  hlasov za SMER;
- Niektore vztahy vypadaju aj kontraintuitivne. Napriklad, ze zvysenie podielu
  studentov vysokych skol v okrese ma silny vplyv na zvysenie podielu hlasov
  za kresťanskodemokratické hnutie; 
- Tiez medzi tymito parametrami chybaju niektore dolezite vztahy, ktore by sme
  ocakavali, napriklad, nikde nevystupuje priemerna mzda, aj ked existuje nazor,
  ze regiony s vyssim platom su viac liberalne, ale s nizsim platom su viac
  konzervativne, co by malo mat suvis s vyberom stran a vysledkami volieb.

To sa da vysvetlit tym, ze model pozera iba na data a relativne rozdiely medzi
okresmi a vysledkami volieb v nich, nie na realnu logiku - model nevie kauzalitu.

Tiez si mozeme vsimnut, ze pre vsetky parametre okresov extremne hodnoty vzdy
maju etnicke politicke strany (Maďarská komunitná spolupatričnosť a MOST-HÍD).
To sa da vysvetlit tym, ze tieto strany takmer nedostali hlasy z okresov, ktore
nie su na juhu Slovenska, kde je primarne lokalizovana madarska mensina.
Logisticka regresia sa pre tieto strany naucila odlisovat juzne okresy od
inych, preto velmi ostro reaguje na zmenu koeficientov. 

**Najdolezitejsie atributy pre politicke strany**

Ak pozrieme na to, ake parametre najviac ovplyvnuju podiel hlasov za
jednotlive politicke strany v logistickej regresie, tiez si vsimneme, ze tie
dolezite atributy vacsinou nezodpovedaju realnej logike:

| Vybrate politicke strany | najdolezitejsi koeficient a parameter       | 2. najdolezitejsi koeficient a parameter          | 3. najdolezitejsi koeficient a parameter          |
|:-----------------|:----------------------------------------------------|:--------------------------------------------------|:--------------------------------------------------|
| PS-SPOLU         | -0.05 (podiel študentov vysokych škol)              | 0.047 (podiel pracujúcich dôchodcov)              | 0.041 (podiel príjemcov kapitálových príjmov)     |
| ĽSNS             | 0.116 (podiel Slovakov)                             | 0.091 (podiel obyvatelov narodenych v SR)         | -0.052 (pocet obyvatelov s iným pracovným stavom) |
| KDH              | 0.18 (podiel študentov vysokych škol)               | 0.129 (podiel Slovakov)                           | 0.086 (podiel žiakov strednej školy)              |
| MOST-HÍD         | -0.311 (podiel Slovakov)                            | 0.133 (podiel obyvatelov s iným pracovným stavom) | 0.107 (podiel obyvatelov v produktívom veku)      |
| MKÖ-MKS          | -0.387 (podiel Slovakov)                | -0.182 (podiel obyvatelov s nezistenou štátnou príslušnosťou) | 0.178 (podiel obyvatelov s iným pracovným stavom) |
| OĽANO a priatela | 0.071 (podiel obyvatelov s iba zakladnym vzdelanim) | -0.051 (podiel obyvatelov narodenych v SR)        | 0.05 (pocet obyvatelov s iným pracovným vzťahom)  |
| SMER             | -0.126 (podiel obyvatelov s iba zakladnym vzdelanim)| 0.087 (podiel Slovakov)                           | 0.061 (pocet Slovakov)                            |
| SaS              | -0.046 (podiel obyvatelov s iba strednym vzdelanim bez maturity) | 0.04 (podiel osob na rodičovskej dovolenke) | 0.038 (podiel osob na materskej dovolenke) |

Opat existuju take vztahy medzi parametrami a volbou stran, ktore da sa pochopit,
napriklad, podiel a pocet Slovakov a nacionalisticke politicke strany, pricom
aj pro-slovenske (ĽSNS), aj pro-mensinske (MKÖ-MKS a MOST-HÍD). Ale vo vascine
pripadov vyber parametrov modelom sa neda vysvetlit. Je mozno neuveritelne, ze
podiel studentov vysokych skol, co vzdy bola najliberalnejsia a najopozicnejsia
skupina, negativne ovplyvnuje podiel hlasov za Progresivne Slovensko, ale
podiel pracujucich dochodcov, co su casto konzervativni obcania, ma vplyv
pozitivny. Je mozno provokativne, ze sa podiel hlasov za opozicnu stranu OĽANO
relativne zvacsi, ak sa zvacsi podiel nevzdelanych obcanov, hoci podiel hlasov
za konzervativnu a proti-europsku stranu SMER pritom zmensi.

Zrejme to nic nehovori o kauzalite. Model bol trenovany na relativne malom pocte
dat, cize neznamena, ze perfektne reflektuje realitu, aj ked dava dobre vysledky.
Tiez su dobre spomenut, ze atributy su casto velmi korelovane a niekedy su aj
zavisle, preto model moze rozhodnut zvazit nejaky parameter viac ako iny, aj ked
v realnom zivote ten iny moze byt viac dolezity.

Takze nie je dobre na zaklade tohoto vysledku hovorit o realnych suvisoch
politickych stran a parametrov. Tiez si dobre uvedomovat, ze koeficienty maju
roznu hodnotu, a -0.05 pri podiele studentov pre Progresivne Slovensko
pravdepodobne ma ovela mensi vplyv ako -0.387 pri podiele Slovakov pre MKÖ-MKS.

**Vplyv na podiel hlasov**

Priama analyza toho, ako zmena parametra na vstupe meni predikciu modelu, je
komplikovana:
- kvoli standartizacie trenovacie aj vstupne data menia svoj tvar, co ma vplyv
  na koeficienty,
- na rozdiel od linearnej regresie, kde linearna zmena koeficienta dava
  linearnu zmenu vysledka, v nasom modeli zavislost nie je linearna
- aby previest logity na pravdepodobnosti, model pouziva softmax, co znamena,
  ze zmena jedneho parametra bude mat vplyv na cely vektor odpovede, pretoze sa
  meni sucet v menovateli.

Preto bolo rozhodnute pozriet na vplyv takym sposobom, ze pre testovy vstup
zmenime niekolko parametrov a pozrieme, ako sa pri tom zmeni predikcia nasho
modelu. Ako kandidaty na zmenu boli vybrate podiel Slovakov v okrese a podiel
studentov vysokych skol v okrese, pretoze tieto atributy maju najvacsi vplyv na
cely model a velky vplyv na predikcie pre jednotlive politicke strany.

Teda po zmene parametrov sme dostali:

| Politicke strany | nezmene | dvakrat viac podiel studentov | dvakrat menej podiel studentov | dvakrat viac podiel Slovakov | dvakrat menej podiel Slovakov |
|:-----------------|--------:|------------------------------:|-------------------------------:|-----------------------------:|------------------------------:|
| PS-SPOLU         |   0.145 |                         0.099 |                          0.170 |                        0.112 |                         0.159 |
| ĽSNS             |   0.047 |                         0.034 |                          0.054 |                        0.072 |                         0.037 |
| KDH              |   0.057 |                         0.137 |                          0.036 |                        0.093 |                         0.043 |
| MOST-HÍD         |   0.005 |                         0.003 |                          0.006 |                        0.000 |                         0.014 |
| MKÖ-MKS          |   0.003 |                         0.001 |                          0.004 |                        0.000 |                         0.010 |
| OĽANO a priatela |   0.253 |                         0.262 |                          0.241 |                        0.229 |                         0.257 |
| SMER             |   0.122 |                         0.120 |                          0.119 |                        0.155 |                         0.105 |
| SaS              |   0.123 |                         0.102 |                          0.131 |                        0.109 |                         0.126 |
| ine              |   0.245 |                         0.242 |                          0.239 |                        0.230 |                         0.249 |

Zvysovanie podielu studentov najviac vplyva pozitivne na KDH - po dvojnasobnom
zvacseni podielu studentov predikcia podielu hlasov za KDH je o takmer 2.5 krat
vacsia, pricom po dvojnasobnom zmenseni je predikcia len o 1.5 krat mensia.

Najhorsie to vplyva na strany s narodnostnou politikou: ak podobne zvacsime
podiel studentov vysokych skol, tak predpovedany podiel hlasov za MOST-HÍD sa
zmensi o 1.7 krat, ale podiel hlasov za Maďarsku komunitnu spolupatričnosť
klesne skoro trojnasobne. Ak o dva krat zmensime podiel studentov, podiel
hlasov za MOST-HÍD sa zvacsi iba o 1.25 a podiel za MKÖ-MKS iba o 1.6.

Zaujimave je Progresivne Slovensko, lebo dvojnasobne zvacsenie podielu
studentov meni vysledok volieb tak, ze podiel hlasov za nich 1.5-krat mensi.
Ako uz bolo spominane, toto je prekvapive spravanie, kedze studenty su casto
liberalne a opozicne.

Pri zmene podielu Slovakov etnicke politicke strany pocituju velmi negativny
vplyv. Ak podiel Slovakov v okrese bude len dvakrat vacsi, podiel hlasov za
MOST-HÍD klesne o 9.5 krat a podiel hlasov za MKÖ-MKS klesne patnastnasobne,
pricom ak sa nastane opacna situacia, podiel hlasov sa stupne trojnasobne pre
MOST-HÍD a takmer stvornasobne pre Maďarsku komunitnu spolupatričnosť.

Tiez je zrejme a ocakavatelne, ze po zvacseni podielu Slovakov v okrese sa zvacsi
podiel hlasov za konzervativne strany: Kotlebovci dostanu o 1.5 krat viac hlasov,
KDH o 1.6, SMER o 1.26. Dolezite si pritom uvedomit, ze po zmenseni koncentracie
Slovakov podiel hlasov za tie strany neklesne prudko - o okolo 1.3 krat pre ĽSNS
a Hnutie a 1.16 pre SMER.

Tiez si mozeme vsimnut, ze ostatne politicke strany pri tychto zmenach
zachovavaju takmer ten isty podiel hlasov, cize pre ne tieto atributy v modeli
nie su velmi dolezite (co potvrdzuje cast o najdolezitejsich atributoch pre
jednotlive strany).

---

### Zhrnutie

- V tejto casti projektu bol urobeny model, ktory predikuje podiely hlasov
  politickych stran v okrese na zaklade socio-demografickych a ekonomickych
  parametrov toho okresu.
- Ako model bola zvolena multinomicka logisticka regresia, ktora sa ukazala ako
  vhodna pre tuto ulohu a dobre predikovala vysledok volieb v okrese.
- Napriek komplikovanym vztahom vstupnych dat a koeficientov modelu, sa podarilo
  najst parametre okresov, ktore maju najvacsi vplyv na vysledok volieb a
  potvrdit ich vyznam a dolezitost pre predikovanie. Hlavnymi atributami sa
  prejavili podiel Slovakov v okrese a podiel studentov vysokych skol v okrese.
- Najdene parametre boli casto zle interpretovatelne kvoli korelacie premennych
  a tomu faktu, ze model pozeral iba na data, nie na realnu socialnu logiku.
  Preto nie je dobre pouzivat takyto nastroj na hladanie kauzality a vztahov v
  realnom svete, aj ked on ma vysoku presnost predikcie.



## Part 3 — Turnout model (Volebná účasť)

### Cieľ analýzy
V tejto časti  sa zameriavame na analýzu faktorov, ktoré ovplyvňujú
volebnú účasť v parlamentných voľbách na úrovni okresov Slovenskej republiky.
Cieľom nie je presná predikcia, ale pochopenie základných vzťahov medzi
charakteristikami okresov a volebnou účasťou.

---

### Premenné

**Závislá premenná**
- `turnout` – volebná účasť v percentách

**Vysvetľujúce premenné**
- demografické: podiel obyvateľov vo veku 0–14 rokov, 65+ rokov a podiel žien,
- trh práce: miera nezamestnanosti (vrátane kvadratického člena),
- urbanizácia a veľkosť okresu: podiel mestského obyvateľstva a logaritmus populácie,
- vzdelanie: podiel vysokoškolsky vzdelaných osôb a osôb bez vzdelania,
- ekonomika: priemerná mzda,
- sociálna štruktúra: podiel cudzincov.

---

### Metódy
Ako hlavný nástroj bola použitá **lineárna regresia (OLS)**, ktorá umožňuje
jednoduchú interpretáciu výsledkov.
Keďže medzi premennými existuje silná korelácia, bola vypočítaná aj
VIF štatistika.

Na výber najdôležitejších premenných bola použitá **LASSO regresia**.
Na základe jej výsledkov sme odhadli aj zjednodušený (restricted) OLS model.
Ako doplnková kontrola bol použitý **Random Forest**, ktorý slúžil len na
porovnanie dôležitosti premenných.

---

### Výsledky
Plný OLS model dosahuje hodnotu \(R^2 = 0,82\) (upravené \(R^2 = 0,78\)),
čo znamená, že vysvetľuje veľkú časť rozdielov vo volebnej účasti medzi okresmi.
Model je ako celok štatisticky významný (F-test, \(p < 0,001\)).

Z výsledkov vyplýva niekoľko hlavných zistení:

- **Vzdelanie má silný vzťah k volebnej účasti.**  
  Podiel vysokoškolsky vzdelaných obyvateľov (`edu_uni`) má
  pozitívny a štatisticky významný vplyv na účasť
  (koeficient približne \(+0,75\), \(p < 0,001\)).
  Naopak, podiel osôb bez vzdelania (`edu_none`) má výrazný
  negatívny vplyv (koeficient približne \(-10,8\), \(p ~ 0,02\)).
  Aj keď ide o malý podiel populácie, táto premenná pravdepodobne
  zachytáva mieru sociálnej marginalizácie v okrese.

- **Nezamestnanosť** je spojená s nižšou volebnou účasťou.
  Lineárny efekt miery nezamestnanosti je negatívny a štatisticky významný
  (koeficient približne \(-1,3\), \(p ~ 0,01\)).
  Kvadratický člen (`unemployment_sq`) má pozitívny, ale štatisticky
  nevýznamný koeficient, čo naznačuje, že vzťah nemusí byť úplne lineárny.

- **Urbanizované okresy** vykazujú nižšiu volebnú účasť.
  Premenná `urban` má stabilne negatívny a vysoko významný efekt
  (koeficient približne \(-0,14\), \(p < 0,001\)).

- **Podiel cudzincov** v okrese je spojený s nižšou účasťou
  (koeficient približne \(-1,6\), \(p ~ 0,01\)).

- **Priemerná mzda** (`avg_wage`) nemá po zohľadnení ostatných premenných
  štatisticky významný samostatný vplyv na volebnú účasť
  (\(p > 0,3\)).

Diagnostika multikolinearity (VIF) ukazuje vysoké hodnoty pre viaceré
socio-demografické premenné, čo je pri tomto type dát očakávané.
Preto boli výsledky ďalej overené pomocou výberu premenných.

Zjednodušený (restricted) OLS model, ktorý obsahuje najstabilnejšie
premenné, dosahuje mierne nižšiu, ale stále vysokú vysvetľovaciu schopnosť
(\(R^2 = 0,76\)).
Zachováva pritom rovnaké základné vzťahy:
pozitívny vplyv vysokoškolského vzdelania a negatívny vplyv
nezamestnanosti, podielu osôb bez vzdelania, urbanizácie a podielu cudzincov.

Ako doplnková kontrola bol použitý model Random Forest.
Poradie dôležitosti premenných potvrdzuje výsledky lineárnych modelov:
najvýznamnejšie sú `edu_none`, `unemployment` (vrátane kvadratického člena),
`foreigners` a `edu_uni`.
To naznačuje, že hlavné závery nie sú závislé od jednej konkrétnej
modelovej špecifikácie.

---

### Poznámky
Medzi použitými premennými je prítomná silná multikolinearita, čo je pri
socio-demografických údajoch bežné.
Preto výsledky interpretujeme hlavne na základe ich stability naprieč
viacerými modelmi.

Zaujímavým výsledkom je silný negatívny efekt pre premennú „bez vzdelania“.
Aj keď ide o malý podiel populácie, pravdepodobne ide o ukazovateľ sociálnej
marginalizácie, ktorá súvisí s nižšou politickou participáciou.

Výsledky naznačujú, že volebná účasť v okresoch SR je ovplyvnená najmä
vzdelanostnou štruktúrou obyvateľstva a situáciou na trhu práce.
Ekonomické ukazovatele majú skôr doplnkový význam.

Táto časť projektu ukazuje, že aj jednoduché regresné modely môžu poskytnúť
zmysluplný pohľad na regionálne rozdiely vo volebnej účasti.
