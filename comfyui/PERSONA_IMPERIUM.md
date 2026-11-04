# AI modelky: od jednej persony k „impériu“

Značky istoty: **[Certain]** overené (kód, oficiálne zdroje), **[Likely]** silný záver,
**[Guessing]** odhad. Čísla o zárobkoch pochádzajú väčšinou od samotných agentúr a blogov,
nikto ich nezávisle neoveril.

---

## 1. Výrobná linka (ComfyUI workflowy 04 → 07)

| Krok | Workflow | Čo robí | Cena za Run [Certain, z kódu ComfyUI] |
|---|---|---|---|
| 1 | `04_persona_1_fuzia_seedream` | telo (foto 1) + tvár (foto 2) → prvá fotka persony | ~0.04 $ |
| 2 | `05_persona_2_referencna_sada_seedream` | 1 fotka → 4 fotky tej istej osoby z rôznych uhlov | ~0.04 $ za každý obrázok (max ~0.14 $) |
| 3 | `06_persona_3_nove_fotky_seedream` | identity pack (3 fotky) + SCENE → nová fotka | ~0.04 $ |
| 4 | `07_persona_4_video_kling` | fotka → 5 s video so zvukom (Kling 4.0) | 720p ~1.12 $, 1080p ~1.68 $ |

**„Pamäť“ persony** tvoria dve veci, ktoré po vytvorení už **nemeníš**:
1. **Identity pack:** 3 najlepšie fotky z kroku 2 (tvár zblízka, 3/4, celá postava).
2. **Persona karta:** text v sekcii `PERSONA:` vo workflowe 06 + karta nižšie.

[Likely] Seedance (video od ByteDance) realistické ľudské tváre ako referenciu odmieta.
Pre reálnych ľudí vyžaduje overenie tváre (liveness), a pre vymyslenú AI tvár ho v ComfyUI urobiť nemôžeš.
Preto video robí Kling.

---

## 2. Persona karta (vyplň pre každú modelku)

```text
Meno / handle:           @________
Vek (vždy 18+):          __ rokov
Odkiaľ je / kde „žije“:  ________
Výzor (PEVNÉ detaily):   vlasy ___, oči ___, výška/postava ___, znamienka/tetovania ___
Podpisový prvok:         (1 vec, podľa ktorej ju spoznáš: farba vlasov, okuliare, šperk…)
Štýl obliekania:         ________
Niche (1 hlavná téma):   fitness / móda / cestovanie / beauty / gaming / lifestyle …
Osobnosť (3 slová):      ________
Záujmy a hobby:          ________
Tón písania:             (vtipná / elegantná / priateľská …), jazyk ___
Čo NIKDY nerobí:         ________
Content piliere (3–5):   1) ____ 2) ____ 3) ____
```

---

## 3. Čo robia najúspešnejšie AI influencerky

| Persona | Kto za ňou stojí | Model zárobku | Poučenie |
|---|---|---|---|
| **Aitana López** | The Clueless (Barcelona) | brand dealy; agentúra hovorí o 6–8 tis. $ za post a 3–10 tis. € mesačne [Guessing, údaje agentúry] | Vedú ju „ako reálneho človeka“: presne daná osobnosť, hobby, výrazný podpisový vzhľad (ružové vlasy). |
| **Lil Miquela** | Brud → Dapper Labs | módne značky, hudba; odhady 8–15 mil. $/rok sa líšia [Guessing] | Príbeh a „drama“, nie len pekné fotky. Má vlastný naratív. |
| **Lu do Magalu** | Magalu (retailer) | persona vlastnená značkou, e-commerce | B2B: značky chcú vlastnú personu. Ako štúdio im ju môžeš vyrobiť. |
| **Emily Pellegrini** | anonymný tvorca | predplatné na Fanvue, uvádza sa >10 tis. $/mesiac [Guessing] | Priamy príjem od fanúšikov. |
| **Shudu, Imma, Noonoouri** | štúdiá (The Diigitals, Aww Inc.) | high-fashion kolaborácie | Silná estetika a konzistentný vizuál pre luxusné značky. |

**Spoločné vzorce** [Likely]:
1. **Jedna jasná niche a jeden podpisový vizuálny prvok.** Nie „pekné dievča“, ale konkrétny charakter.
2. **Osobnosť a príbeh:** majú názory, hobby a „život“ (cestovanie, kamaráti, výzvy).
3. **Absolútna konzistencia tváre.** Jedna nezhoda a publikum stratí dôveru.
4. **Model štúdia:** The Clueless vlastní postavy a licencuje ich značkám. Predáva lacnejšie než ľudskí influenceri a obsah je dostupný 24/7.

---

## 4. Ako z toho spraviť „impérium“ (postup)

1. **Fáza 1 (0–90 dní): 1 persona.** Vyplň kartu, vyrob identity pack a 30 fotiek do zásoby.
   Publikuj 4–5 postov a 3–5 Reels/TikTokov týždenne. Sleduj, čo funguje (uloženia a zdieľania, nielen lajky).
2. **Fáza 2: monetizácia prvej persony.** Affiliate odkazy, menšie brand dealy (kontakty v bio, media kit),
   prípadne predplatné.
3. **Fáza 3: druhá persona v inej niche.** Rovnaká linka 04 → 07, nová karta a nový identity pack.
   Každá persona má vlastný priečinok `persona_MENO/` (identity pack, karta, kalendár obsahu).
4. **Fáza 4: štúdio.** Ponúkaj značkám ich vlastnú personu alebo licenciu na tvoje (model Lu do Magalu / The Clueless).

[Guessing] Prvé tri mesiace rozhodnú, či persona chytí. Väčšina AI influencerov veľa nezarába,
top príklady sú výnimky.

---

## 5. Pravidlá, ktoré ti môžu účet zabiť (nie sú voliteľné)

- **Označenie AI:** [Likely] Instagram má profilový atribút *AI-generated profile*.
  Neoznačené AI účty môže obmedziť v dosahu. [Likely] TikTok od 9/2026 vyžaduje AIGC label na realistickom AI obsahu,
  inak hrozí odstránenie. Do bio daj napr. „AI-generated virtual creator“.
- **Reklamy:** [Likely] V USA (zákon v New Yorku, FTC) musí platená spolupráca s AI personou mať označenie AI **aj** #ad.
- **Žiadne reálne tváre bez súhlasu.** Fotka 2 vo fúzii musí byť vymyslená AI tvár, tvoja tvár, alebo tvár človeka s písomným súhlasom.
  Nikdy nie celebrita ani cudzí človek z Instagramu.
- **Persona je vždy dospelá (18+)** a nevydáva sa za reálnu existujúcu osobu.
- **Žiadni boti a kúpení sledujúci.** Platformy to odhalia a znížia dosah.
- **Obsah pre dospelých:** [Likely] OnlyFans vyžaduje overeného reálneho človeka, Fansly realistické AI zakázal,
  Fanvue AI tvorcov povoľuje, ale vyžaduje označenie a overenie prevádzkovateľa. Pred štartom si over aktuálne podmienky.

---

## Zdroje
- [Fake influencers, real money: Inside the AI creator business (Yahoo Tech)](https://tech.yahoo.com/ai/articles/fake-influencers-real-money-inside-203551126.html)
- [She has 400,000 Instagram followers and major brand deals. She's also AI (Fast Company)](https://www.fastcompany.com/91546466/she-has-400000-instagram-followers-and-major-brand-deals-shes-also-ai)
- [Aitana López (Wikipedia)](https://en.wikipedia.org/wiki/Aitana_L%C3%B3pez)
- [The Clueless: Accelerating virtual influencer growth (MIDiA Research)](https://www.midiaresearch.com/case-studies/the-clueless-agency-accelerating-virtual-influencer-growth)
- [Richest AI Influencers: The 2026 Gold Rush](https://blog.ntice.app/richest-ai-influencers/)
- [Virtual Influencers in 2026: 15 Real Examples (Ryla)](https://ryla.ai/en/blog/virtual-influencer-examples/)
- [Instagram Demotes Unlabeled AI Influencer Accounts](https://letsdatascience.com/news/instagram-demotes-unlabeled-ai-influencer-accounts-0cbd924b)
- [AI influencer disclosure rules by platform (Oakgen)](https://oakgen.ai/blog/ai-influencer-disclosure-rules-by-platform)
- [New York AI Influencer Law (Common Thread)](https://commonthreadco.com/blogs/coachs-corner/new-york-ai-influencer-disclosure-law-ecommerce)
- [Fanvue at $65M ARR (Sacra)](https://sacra.com/research/fanvue-at-65m-arr/)
- [Seedance 2.0 asset & face reference FAQ (APIYI)](https://docs.apiyi.com/en/faq/seedance2-asset-face-reference)
- [Seedance 2.0 Real Human – ComfyUI docs](https://docs.comfy.org/tutorials/partner-nodes/bytedance/seedance-2-0-real-human)
