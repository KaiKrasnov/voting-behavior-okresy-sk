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

### Diskusia
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
