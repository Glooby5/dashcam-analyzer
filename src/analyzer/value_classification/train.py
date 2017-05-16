import argparse
import cv2
import numpy as np
import sys
import os.path


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset file", required=True)
args = vars(ap.parse_args())

responses = []
samples = np.empty((0, 100))

with open(args['dataset'], "r") as ins:
    for line in ins.readlines():
        dir_class = line.rstrip().split(':')[0]
        path = line.rstrip().split(':')[1]

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image is None:
            continue

        roi_small = cv2.resize(gray, (10, 10))

        sample = roi_small.reshape((1, 100))
        samples = np.append(samples, sample, 0)
        responses.append(dir_class)

print("training complete")

np.savetxt('symbol-classification-samples.data', samples)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
np.savetxt('symbol-classification-responses.data', responses)

# %=========================================================================
# % (c) Michal Bidlo, Bohuslav Křena, 2008
#
# \chapter{Úvod}
#
# Veřejné pozemní komunikace jsou lemovány velikým množstvím dopravních značek, získat informace o jejich pokrytí je však problematické. Vědět, kde se nachází jaká dopravní značka je důležité například pro navigační systémy, ale může to být zajímavá informace pro veřejnou správu nebo i běžné občany. Proto se tato práce zabývá analýzou záznamu z palubní kamery, kdy se snaží všechny tyto informace z pořízeného záznamu získat.
#
# Cílem práce je při využití mobilního telefonu nebo jiného záznamového zařízení se schopností záznamu videa a informací o GPS poloze analyzovat pořízené videozáznamy a to za cílem zmapování pokrytí dopravního značení podél pozemních komunikací. Za tímto účelem jsou v práci navrhnuty postupy jak je možné detekovat zákazové dopravní značení z pořízeného záznamu. Dále jsou představeny postupy jak určit konkrétní typ dopravní značky, tak aby bylo možné detekované objekty třídit podle skupin. V případě detekce rychlostního omezení je dále navržena možnost klasifikace jeho hodnoty, tak aby informace měla přesný význam. Tyto vymezené cíle vedou k tomu jak z nich vytvořit souhrné informace, které budou mít vypovídající hodnotu pro cílovou skupiny.
#
# \todo{Popsat co se řeší v jednotlivých kapitolách?}
#
#
# \chapter{Existující řešení}
#
# Tato kapitola se zabývá tím jaké existují použité postupy pro detekci, klasifikaci typu či hodnot dopravních značek ze záznamu či online videa pořizovaného z jedoucího automobilu.
#
#
# \section{Detekci dopravních značek v obraze}
#
# Základním bodem této práce je detekce dopravní značek, proto jsou  zde rozebrány  existující řešení, které se zabývají touto problematikou a představují různé postupy řešení. Pro detekci dopravního značení existují různé metody odlišující se různou složitostí nebo efektivitou.
#
# \subsection{Metody založené na detekci tvarů}
#
# Jednou ze základních metod  je zpracování obrazu s využitím detekce tvarů pomocí Houghovy transformace \cite{1706843}. V odkazované práci jsou v binární reprezentaci vyhledávány objekty specifického tvaru, které pravděpodobné odpovídají dopravním značkám. Výhodou detekce tímto postupem je schopnost detekce i částečně zakrytých značek.
#
# Metodám založeným na detekci tvaru často předchází segmentování oblastí podle barvy, tak aby byl značně zmenšen prostor pro vyhledávání \cite{5378370}. Využívá se toho, že dopravní značky jsou reprezentovány specifickými a většinou výraznými barvami. Ve vstupní barevném obrázku jsou tedy vyhledávány oblasti v rozsahu zadané barvy. To probíhá tak, že se pro každý bod obrazu kontroluje zda jednotlivé složky barvy spadají do uvedeného intervalu.
#
# Vstupní obraz je ve většině případů v RGB modelu, ten sice umožňuje na základě rozsahu například červené barvy segmentovat požadované oblasti, ale pro tyto účely je vhodnější převést vstupní snímek do modelu HSV \cite{mohdali2013}. To sice vede k nepatrně vyšší náročnosti na výpočetní výkon, ale úspěšnost segmentace je mnohem vyšší. To je dané tím, že tento typ barevného modelu není tak citlivý na změnu osvětlení. Možnost nastavit konkrétní hodnoty pro sytost a jas je právě to v čem vyniká. V uvedené práce jsou srovnávány úspěšnosti segmentace pro různé vzorky. Úspěšnost je vyhodnocována pomocí následujícího vzorce
#
# $$a_{d} = \frac{b_{sd}}{b_{td}} \times 100$$
#
# kde $b_{sd}$ odpovídá počtu správných detekcí a $b_{td}$ celkovému počtu snímků. V případě značek červené barvy je při částečné okluzi rozdíl velmi markantní, pro červenou značku v RGB modelu je úspěšnost 51\,\%, za to pro HSV model dosahuje úspěšnost 77\,\%. Při nulové okluzi je pro červenou barvu rozdíl 10\,\% opět ve prospěch HSV.
#
#
# \subsection{Neuronové sítě}
#
# Další možností jak detekovat dopravní značení je využití konvolučních neuronových sítí \cite{6706811}. Využití konvolučních neuronových sítí je v oblasti detekce čím dál častější, proto se autoři citované práce pokusili aplikovat tuto metodu na detekci dopravních značek.
#
# Podobně jako v předchozím případě je vstupní obraz nejprve různými způsoby předzpracován. Tím podstatným je to, že autoři navrhují jak co nejlépe převést vstupní barevný obraz do šedotónového a zároveň omezit oblasti možné detekci na základě analýzy barevného modelu. Na rozdíl od přechozích popsaných řešení k tomu nevyužívají přímo analýzu například nad modelem HSV, ale navrhují použít SVM klasifikátor. Ten bude natrénován, tak aby byl schopen určit jestli intenzita pixelu spadá do požadovaného odstínu barvy. Cílem tohoto přístupu je jako v předcozím případě zvýšit úspěšnost segmentace za různých světelných podmínek.
#
# Nad připraveným obrazem je spuštěna konvuluční neuronová síť, která je natrénovaná pro detekci několika typů dopravní značky. Výsledky detekce uvedené v práci dosahují velmi vysokých hodnot a to až přes 95\,\%. Ale jak sami autoři zmiňují největší slabinou tohoto řešení je časová náročnost, které i přes předzpracování obrazu je velmi vysoká.
#
# \section{Klasifikační metody}
#
#
# \subsection{Porovnávání šablon}
#
#
# \subsection{SVM klasifikátor}
#
#
# \subsection{Neuronové sítě}
#
#
#
# \chapter{Použité algoritmy}
#
#
# \section{Kaskádový klasifikátor}
# \label{sec:cascasdeClassifier}
#
# Pro detekci dopravní značek je v této práci zvolen kaskádový klasifikátor v jehož použití jsou hlavními průkopníky dvojice Viola a Jones \cite{990517}. Ti použili kaskádový klasifikátor pro detekci obličejů v obrazu a dosáhli velmi vysoké úspěšnosti přesahující 90\,\%. Autoři práci staví na několika základních prvcích umožňující dosáhnout uvedených výsledků.
#
# Základem je omezení analýzy na základě jednotlivých bodů obrazu, ale využití příznaků zahrnující širší oblast zájmu. To umožňuje získat informace, které při analýze po samostatných pixelech chybí, ale hlavním dopadem je rapidní zvýšení rychlosti. Autoři využívají příznaků, které se podobají těm Haarovým. Aby je bylo možné vypočítat, je vstupní snímek převeden na reprezentaci integrálního obrazu. Ten lze vyjádřit pomocí následujícího vzorce, kde aktuální pixel odpovídá součtu pixelu nad a vlevo od aktuálního pixelu:
#
# \begin{equation}
# ii(x,y) = \sum_{x^{'}\leq x, y^{'}\leq y} i(x^{'},y^{'})
# \end{equation}
#
# kde $ii(x,y)$ představuje integrální obraz a $i(x,y)$ odpovídá originálnímu obrazu. Pomocí této reprezentace, lze snadno vypočítat obdélníkové příznaky na kterých tento princip staví.
#
# Druhým podstatným bodem je využití algoritmu AdaBoost. Počet příznaků je v obraze velmi vysoký a proto je důležité se zabývat pouze těmi výraznými, tak aby bylo možné při klasifikaci postupovat co možná nejefektivněji a soustředit se na malý soubor příznaků. To je dosaženou modifikací algoritmu AdaBoost, tak aby každý slabý klasifikátor závisel pouze na jednom příznaku. Výsledkem každé úrovně procesu posilování je výběr nového slabého klasifikátoru, což odpovídá selektivní funkci procesu.
#
# Třetím základním rysem tohoto postupu je použití série slabých klasifikátorů, které dohromady vytváří kaskádu. Tento princip zakládá na tom, že na začátku jsou zařazeny menší klasifikátory jejichž úkolem je odstínit co největší počet negativních detekcí. Díky tomu je možné za ně zařadit klasifikátory, které jsou výpočetně náročnější, ale díky tomu, že je výrazně snížen počet negativních detekci, tak je jejich úkolem odstínit už jenom zbývající procento nechtěných oblastí, tak aby na konci série klasifikátoru byly pouze pozitivní detekce. Na schématu ~\ref{fig:CascadeClassifierScheme} je znázorněn postup, kdy jsou na vstupu všechna podokna v rámci vstupního obrazu, která jsou postupně vyřazována jako negativní až na konci zůstanou ty, které by měly odpovídat pozitivním detekcím.
# % * <kaderabek.jan@gmail.com> 2017-05-08T12:10:36.278Z:
# %
# % ^.
#
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.6\linewidth,]{obrazky-figures/cascade-classifier.pdf}\\[0.5pt]
# \caption{Schéma kaskádového klasifikátoru znázorňující sériové zařazení slabých klasifikátorů, které postupně vylučují negativní detekce}
# \label{fig:CascadeClassifierScheme}
# \end{figure}
#
#
# \subsection{LBP}
# \label{sec:lbp}
#
# Tato práce pro použití kaskádového klasifikátoru nevolí standardní Haarovy příznaky, ale příznaky typu LBP \emph{(Local Binary Patterns)}. A to z toho důvodu, že jejich použití je mnohem efektivnější a jejich použití výrazně zvyšuje rychlost klasifikátoru jak dokazuje porovnání \cite{7006273}. V odkazované práci přináší použití LBP pro detekci obličejů až dvoůapůlnásobné zrychlení.
#
# \begin{figure}[H]
# \centering
# \includegraphics[width=0.6\linewidth,]{obrazky-figures/lbp.pdf}\\[0.5pt]
# \caption{Postup získání hodnoty příznaku LBP prahováním okolním hodnot přes prostřední pixel}
# \label{fig:LBPScheme}
# \end{figure}
#
# Proces získání hodnoty příznaku je ilustrován obrázkem ~\ref{fig:LBPScheme}. Pro určení hodnoty příznaku LBP slouží zpracovávaný bod a jeho přímé okolí, které je tvořeném okolními osmi pixely. Vzniká tak matice $3 \times 3$, kde prostřední pixel slouží jako práh s nímž jsou porovnávány ostatní hodnoty z matice \cite{OJALA199651}. Během prahování se porovná intenzita pixelu s hodnotou prahu a vytváří se tak matice s binární reprezentací. Z takto vytvořené matice je sestaveno binární číslo o osmi bitech, které je vytvořeno zapisováním bitů postupným procházením matice z levého horního rohu po směru hodinových ručiček. Maximální. hodnota příznaku je tedy 256 v dekadické soustavě a aktuální hodnotu získáme převodem z binární reprezentace.
#
# \subsection{AdaBoost}
#
# Klasifikační metoda AdaBoost, který je v této práci zakládá na principech \emph{boosting} strojového učení. Tato metoda umožňuje zlepšit přesnost klasifikaci jakéhokoliv algoritmu pro strojové učení. To je zajištěno tím, že jsou postupně vytvářený slabé klasifikátory, které postupně vytvoří silný klasifikátor s vyšší úspěšností. Výstupem použití této metody je silný lineární klasifikátor $H(x)$, ten je tvořen kombinací slabých klasifikátorů $h(x)$ \cite{Schapire2013}.
#
# Slabé klasifikátory vykazují velmi malé procento úspěšnosti srovnatelné s náhodou. Vstupem je trénovací množina S, složena z dvojic $(x_{i}, y_{i})$, kde i nabývá hodnot 1 až M, kde M určuje velikost množiny S. Během trénování jsou využívány váhy $D_{t}$, které jsou z počátku nastaveny rovnoměrně. V každém průběhu dochází k výběru klasifikátoru, který splňuje podmínku maximální chyby, která by měla mít maximální hodnotu $0,5$ a zároveň je tato chyba nejmenší při určených vahách. Váhy jsou následně aktualizovány pro další proces. V případě splnění podmínek je tento slabý klasifikátor vybrán a pokračuje se na další úroveň. Pokud nejsou splněny podmínky maximální chyby dojde k vyhledání jiného slabého klasifikátoru. Vyhledávání by mělo být úspěšnější díky tomu, že se zvýší váhá špatně klasifikovaného řešení a naopak se sníží váha správně klasifikovaného bude zmenšena. To způsobuje exponenciální snižování chyby při zvyšujícím se počtu slabých klasifikátorů.
#
# \section{k-Nearest Neigbor}
# \label{sec:knn}
#
# Klasifikace podle nejbližších sousedů spadá mezi standardní metody strojového učení s  učitelem. Jedná se o bezparametrickou metodou jejíž princip je založený na  porovnání Euklidovské vzdálenosti  porovnávaných vzorků.  Její předností je to, že její trénovací proces není náročný na čas. Při klasifikaci však musí mít v paměti všechny trénovací vzorky.
#
# Princip klasifikace základní metody Nerarest Neigbour někdy označované také jako 1-NN lze popsat následovným postupem \cite{1053964}. Máme dáno $n$ vzorků $X_{1}, X_{2}, \ldots , X_{n}$ v trénovací sadě $X$ a chceme určit třídu klasifikovaného vzorku $P$. Toho dosáhneme porovnáváním podobnosti vzorku $P$ se všemi vzorky z množiny $X$. Podobnost představuje vzdálenost $d(P, X_{i})$ určující vzdálenost vzorku $P$ od daného vzorku $X_{i}$. Vzorek $P$ patří do té třídy vzorku $X_{k}$, kde platí:
#
# \begin{equation}
# d(P, X_{k}) = min\{d(P, X_{i})\}
# \end{equation}
#
# kde $i = 1 \ldots n$.
#
# Rozšířená metoda standardní klasifikace pomocí nejbližších sousedů neurčuje třídu podle vzorku s nejmenší vzdáleností, ale podle $K$ nejbližších vzorků. Jako příslušná třída klasifikovaného vzorku je určena ta třída, která je mezi $K$ sousedy nejvíce zastoupena. Tento postup umožňuje odstínit situace kdy je jako nejbližší soused klasifikován vzorek, který představuje pouze odchylku své třídy.
#
# \section{CLAHE}
# \label{sec:clahe}
#
# Kontrastem limitovaná adaptivní ekvalizace histogramu \emph{Contrast Limited Adaptive Histogram Equalization (CLAHE)} \cite{min2013novel} je v této práci využívána za účelem  zlepšení výsledků prahování. Cílem ekvalizace histogramu je zvýšení kontrastu za účelem rovnoměrného rozložení jasu, tak aby vynikaly různé detaily vstupního obrazu.
#
# Standardní ekvalizace histogramu není příliš vhodná  jelikož při nerovnoměrném osvětlení nedokáže upravit intenzity v tmavé části obrazu. Z toho důvodu je často používána metoda adaptivní ekvalizace histogramu \emph{(AHE)}. Ta využívá principu, že histogram obrazu je ekvalizován po částech. Hodnota pixelu je nahrazena ekvalizovanou hodnotou z okolí upravovaného bodu. Díky tomu výsledek dosahuje vylepšeného lokálního kontrastu. Tento postup s sebou však nese rizika, že v homogeních oblastech vstupního obrazu se začne objevovat šum.
#
# Právě tento problém řeší použití metody \emph{CLAHE}, která zavádí použití parametru \emph{clip limit}. Ten slouží k limitaci kontrastu. Pokud je tedy lokálním histogramu pixelu detekován jas shodnotou vyšší než je limit, tak se jeho sníží právě na limit. To umožní, že se hodnota nad limitem rovnoměrně rozloží do ostatních částí histogramu.
#
# \section{Adaptivní prahování}
# \label{sec:adaptiveThresh}
#
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.6\linewidth,]{obrazky-figures/threshold.jpg}\\[0.5pt]
# \caption{Obrázky demonstrující rozdíl ve volbě hodnoty prahu}
# \label{fig:ThresholdValue}
# \end{figure}
#
# Pro účely prahování šedotónového obrazu je stejně jako v předchozím případě nevhodná globální použití určité hodnoty prahu. Hodnotu prahu je tedy vhodné volit na základě okolí daného pixelu. Vzorec pro vytvoření binární reprezentace obrazu z té šedotónové pomocí standardního prahování vypadá následovně:
#
# \begin{equation}
# g(x,y) =
#   \begin{cases}
#     1       & \quad \text{pro } f(x, y) > T \\
#     0  & \quad \text{jinde}\\
#   \end{cases}
# \end{equation}
#
# kde $T$ odpovídá hodnotě prahu v rozsahu 0 až 255. Hlavním úskalím tohoto postupu je právě fixní hodnota prahu, která je aplikována globálně na celý vstupní obraz. To vede k tomu, že pro některý typ vstupních vzorků je práh zvolen vhodně, ale pokud například nastavíme práh ve prospěch světlejších obrazů, tak výsledek prahování nebude použitelný pro obrázek s nižším jasem.
#
# Z toho důvodu je nutné volit hodnotu prahu automaticky. To může probíhat podobně jak v předchozí metodě adaptivní ekvlaizace histogramu, kdy je výsledná intenzita pixelu spočítána na základě intenzit okolních pixelů. Upravený vzorec bude mít následující tvar:
#
# \begin{equation}
# g(x,y) =
#   \begin{cases}
#     1       & \quad \text{pro } f(x, y) > T(x,y) \\
#     0  & \quad \text{jinde}\\
#   \end{cases}
# \end{equation}
# kde $T(x,y)$ odpovídá hodnotě prahu spočítaného individuálně pro každý pixel. Metod jak spočítat hodnotu prahu může být více. Jednou z nich může být průměr intenzit okolí pixelu. Velikost okolí určuje hodnota velikosti bloku a standardně může nabývat hodnot $3, 5, 7, \ldots$.
#
# V této práci je pro výpočet aktuálního prahu použita metoda, kdy hodnota prahu odpovídá váženému součtu okolí, kde váhy odpovídají Gaussově oknu \footnote{\url{http://docs.opencv.org/trunk/d7/d4d/tutorial\_py\_thresholding.html}}. Výsledky aplikace jednotlivých metod demonstruje obrázek  ~\ref{fig:ThresholdValue}.
#
#
#
# %%%%%%%  NÁVRH ŘEŠENÍ  %%%%%%%
#
# \chapter{Návrh řešení}
#
# V této kapitole jsou prezentovány postupy jaké jsou zvoleny k řešení vymezených požadavků a to konkrétně detekce dopravní značky v obraze, následná klasifikaci druhu a případné klasifikování zobrazované hodnoty. Výsledkem této části by měl být navržený postup pro kompletní analýzu záznamu palubní kamery automobilu  na jejímž konci jsou dostupné konkrétní informace o pokrytí dopravního značení kolem pozemních komunikací.
#
# \section{Popis systému}
#
# Navržený systém musí umožňovat zpracovat celý videozáznam včetně příslušných informací o poloze vozu. Základní princip je popsán na schématu ~\ref{fig:SystemSchema}. Systém by měl zpracovávat snímky v pořízeném záznamu z palubní kamery a v každém se pokusit detekovat zákazové dopravní značky. Po detekování značky následuje rozhodnutí o jaký konkrétní druh značky se jedná. V případě detekování rychlostního limitu bude systém umožňovat klasifikaci zobrazovaného rychlostního omezení, tak aby detekce takovéto dopravní značky měla vypovídající hodnotu. Pokud bude detektor neúspěšný, ale na snímku by se podle předcházejících informací měla nacházet dopravní značka, tak se provede vypočítání nové pravděpodobné pozice a bude tak umožňovat sledování detekovaných značek v předcházejících snímcích.
#
# Výsledný systém bude umožňovat vytvořit report, který bude zahrnovat unikátní výskyty dopravní značky včetně GPS souřadnic o její pozici na mapě. Proto bude zapotřebí rozhodnout v jaký okamžik dojde k exportování detekovaného objektu a podle čeho se určí pozice dopravní značky.
#
#
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.7\linewidth,]{obrazky-figures/schema-systemu.pdf}\\[1pt]
# \caption{Schéma systému zpracování jednoho snímku záznamu}
# \label{fig:SystemSchema}
# \end{figure}
#
#
# \section{Detekce dopravní značky}
#
# Prvním bodem při zpracování pořízeného videozáznamu je detekce dopravní značky v rámci aktuálně zpracovávaného snímku. K těmto účelům je na základě předchozího rozboru zvolen kaskádový klasifikátor ~\ref{sec:cascasdeClassifier}.  Pro jeho trénování je v následující kapitole popsán postup příprava datové sady. K jeho implementaci je zvoleno použití příznaků LBP ~\ref{sec:lbp}, které umožňují rychlejší proces trénování, jehož výsledkem nemusí být tak přesná detekce jako v případě Haarových příznaků, nicméně v případě detekce zákazových dopravních značek tento fakt nebude hrát výrazný problém díky vlastnostem zmiňovaného typu dopravního značení.
#
# Výsledkem aplikace kaskádového klasifikátoru na snímek jsou tzv. bounding boxy, které představují pravděpodobnou pozici dopravní značky v rámci snímku.  S využitím  získaných souřadnic se provede vyříznutí zájmové oblasti ze vstupního snímku a tyto výřezy budou předány k dalšímu kroku zpracování, kterým je klasifikace typu dopravní značky. Vzhledem k tomu, že proces detekce pomocí kaskádového klasifikátoru není z pohledu využití systémových prostředků příliš levná  záležitost, tak je navrhnuto, že není nutné zpracovávat veškeré snímky předaného videozáznamu, ale bude možné zpracovávat  například každý druhý snímek. To je možné díky tomu, že  i ve vyšších rychlostech není mezi snímky ve videu při třiceti snímcích za sekundu příliš velký rozdíl a  sousední snímky disponují prakticky stejnou hodnotou informací.
#
#
# \section{Klasifikace druhu dopravní značky}
# \label{sec:TypeDesign}
#
# Pro klasifikaci typu dopravní značky je využito zpracování detekované oblasti z předchozího kroku. Proces klasifikace se sestává z několika kroků jak znázorňuje obrázek ~\ref{fig:TypeClassification}.
# Detekovaná oblast dopravní značky nezahrnuje  pouze dopravní značku, ale také část okolí, které je  pro každou detekci jiné, což je považováno za rušivý a nechtěný vliv.  Proto je nutné  detekovanou oblast ještě zúžit a získat, tak pouze výřez, který  obsahuje  co nejméně okolního šumu.  To lze řešit  vyprahováním výběru do binární reprezentace a  pomocí detekce kontur zvolit tu nejvíc vnější, která bude pravděpodobně odpovídat dopravní značce.  Na vstupu se však mohou objevit  vzorky, s nízkým kontrastem mezi okolím a samotnou oblastí dopravní značky.
#
# Aby bylo možné prahováním získat relevantní výsledek, je vhodné nejprve provést ekvalizaci histogramu, tak aby bylo možné od sebe separovat jednotlivé oblasti.  Standardní ekvalizace předpokládá rovnoměrné rozložení intenzit v rámci obrazu, to  by v tomto případě  mohlo vést ke splynutí některých oblastí. Z toho důvodu je navrhnuto využít adaptivní ekvalizaci histogramu ~\ref{sec:clahe} , která se provádí pro každý pixel porovnáním hodnot z jeho okolí. Výsledkem je potom kontrastní obraz ve kterém jsou  zachovány a hlavně zvýrazněny podstatné přechody.  Pro prahování je opět vhodné zvolit adaptivní variantu ~\ref{sec:adaptiveThresh}, tak aby bylo možné respektovat obrazy s různou úrovní jasu.
#
# Nad takto zredukovaným výřezem lze znovu provést ekvalizaci histogramu a prahování, tak abychom ze značky získaly vnitřní piktogramy, které  jsou hlavním příznakem pro určení typu dopravní značky. Klasifikace probíhá  klasifikátorem implementujícím metodu porovnání K-Nearest Neigbour ~\ref{sec:knn}. Jako příznaky pro klasifikaci touto metodou je navrhnuta právě konečná binární reprezentace dopravní značky, která je serializována do vektoru.
#
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.9\linewidth,]{obrazky-figures/klasifikace-typu.pdf}\\[1pt]
# \caption{Postup zpracování detekované oblasti pro získání vektoru příznaků za účelem klasifikace typu dopravní značky}
# \label{fig:TypeClassification}
# \end{figure}
#
# \section{Klasifikace hodnot dopravní značky}
# \label{sec:ValueDesign}
#
# V případě dopravní značky typu rychlostního omezení je samotná informace nepříliš vypovídající a je tak podstatné znát hodnotu omezení. K její klasifikace je opět použita metoda K-Nearest Neigbour, postup  se ale trochu liší od klasifikace typu.  Pro získání hodnoty rychlostního limitu je vhodné klasifikovat číselné symboly samostatně.  Z toho důvodu je nutné nejprve provést  jejich extrahování z výřezu dopravní značky.
#
# K tomuto účelu je  použit upravený výřez dopravní značky, tedy  vybraná pouze  hlavní oblast dopravní značky na kterou je aplikováno adaptivní prahování pro získání binárního obrazu. V tomto obrazu lze provést vyhledávání kontur a vybrání těch, které splňující určitě vlastnosti jako je výška, šířka, celkový počet nastavených pixelů, tak aby bylo možné určit, že se jedná o číselný symbol, který je  ze značky vyříznut. Binární reprezentace opět slouží jako příznaky pro klasifikaci a díky tomu je možné rovnou získat příslušnou hodnotu tohoto znaku. Postup je znázorněn na obrázku ~\ref{fig:ValueClassification}.
#
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.7\linewidth,]{obrazky-figures/klasifikace-hodnoty.pdf}\\[1pt]
# \caption{Postup klasifikace hodnoty rychlostního omezení}
# \label{fig:ValueClassification}
# \end{figure}
#
# Získání  výsledné hodnoty se tedy sestává z postupné klasifikace symbolů, proto je potřeba mít na paměti jaké je pořadí těchto symbolů na vstupním vzorku. Klasifikování po jednotlivých znacích může způsobit to, že například pro dvoucifernou hodnotu bude klasifikace jednoho symbolu správná a pro druhý ne, to ale znamená, že kompletní hodnota není validní a takové případy je potřeba rozpoznat. Nejjednodušším možným řešením je využít faktu, že rychlostní limitu nabývají specifických hodnot. Konkrétně víme, že na území České republiky se objevují limity do hodnoty 130, právě díky tomu lze označit klasifikovanou hodnotu jako nevalidní pokud přesáhne tento limit. Dalším faktem je to, že se v naprosté většině případů jedná o hodnoty dělitelné číslem deset. Pokud je tedy výsledkem dělení klasifikované hodnoty a čísla 10 nenulový zbytek, opět to znamená, že zjištěná hodnota není validní. V některých případech se, ale mohou objevovat i hodnoty dělitelné pěti. Ty se vyskytují v menší míře a záleží na konkrétní implementaci jestli jsou brány v potaz i taková omezení.
#
#
# \section{Sledování pozice dopravní značky}
#
# Pokud víme, že při zpracování předchozího snímku jsou přítomny detekované dopravní značky, které nebyly detekovány na aktuálním snímku, znamená to, že pravděpodobně došlo k výpadku detekce. Aby bylo možné snadněji postupovat při dalším zpracování, je vhodné tyto prázdná místa vyplnit. Tím je myšleno predikovat pravděpodobnou pozici dopravní značky na aktuálním snímku.
#
# Toho lze dosáhnout díky uloženým informacím o pozicích dopravní značky z předcházejících snímků. Pokud máme alespoň dvě předchozí pozice a rozměry je už snadné vypočítat na jakých souřadnicích a s jakými rozměry bude interpretována dopravní značka, kterou se nepovedlo detekovat. Výpočet bere rozdíl souřadnic mezi dvěma předcházejícími snímky a jejich rozdíl přičte k předcházejícím souřadnicím. Obecný vzorec pro vypočítání pravděpodobné pozice $x_{n}$ pro aktuální snímek $n$ lze vyjádřit vzorcem $x_{n} = x_{n - 1} + (x_{n - 1} - x_{n - 2})$. Tento vzorec lze použít díky velké frekvenci snímků a malým rozdílům pozic mezi jednotlivými snímky.
#
# \section{Exportování dopravní značky}
#
# Při procházení záznamu se provádí detekce a klasifikace na všech snímcích. Pro účely exportu souboru shrnujícím pokrytí dopravním značení je vhodné provést export posledního výskytu konkrétní dopravní značky v rámci záznamu. Díky offline zpracování lze jako poslední výskyt prohlásit okamžik, kdy se značka určitého typu nenachází na určeném počtu následujících snímků. Pokud je klasifikace úspěšná může se přejít k uložení. Pokud se ale například nepovedlo nad aktuálním snímkem klasifikovat hodnotu rychlostního omezení je možné se pokusit dohledat ji v rámci předchozích snímků. To stejné platí i pro klasifikaci typu dopravního značení.
#
# Při finálním exportování dopravní značky je nutné k zapisovaným datům dodat, také informace o poloze dopravní značky v rámci mapy. K umožnění tohoto kroku bude požadováno, aby vstupní soubor záznamu byl doplněn také souborem s GPS souřadnicemi v různých časech záznamu. Při ukládání se použije záznam, který nejvíce odpovídá času aktuálního snímku.
#
#
# %%%%%%%  DATOVÉ SADY  %%%%%%%
#
# \chapter{Tvorba datových sad}
#
# Tato kapitola se zabývá popisem postupu vytváření všech potřebných datových podkladů nutných pro implementaci zvolených metod v kapitole návrhu řešení.
#
#
# \section{Pořizování videozáznamů}
#
# Za účelem vytvoření datových sad pro implementaci navrhnutých řešení bylo zapotřebí sbírat záznamy z palubní kamery automobilu. Všechny tyto záznamy byly pořizovány pomocí mobilního telefonu s operačním systémem Android ve verzi 6. Ten disponuje kamerou, která umožňuje pořizovat snímky o maximální velikosti 13\,Mpix.
#
# Aby bylo možné při pozdější analýze záznamu k detekovaným dopravním značkám přiřadit pozici na mapě je nutné mít nejen samotný videozáznam, ale také GPS souřadnice pro aktuální polohu vozidla v konkrétním čase. To neumožňuje žádný nástroj, který mobilní telefon v základní konfiguraci nabízí. Proto bylo nutné nalézt aplikaci třetí strany, která bude splňovat uvedené požadavky.
#
# Jako ideálním kandidátem se ukázala aplikace \emph{AutoBoy Dash Cam - BlackBox}. Ta umožňuje v bezplatné verzi pořizovat záznamy o maximálním rozlišení $1920\times1080$\,px při 30 snímcích za vteřinu. Pro účely v této práce však bude dostačovat rozlišení $1280\times720$\,px, což je rozumný kompromis mezi kvalitou a rychlostí zpracování, která se u vyššího rozlišení značně prodlužuje. Podstatnou věcí je však to, že apliakce ke všem pořízeným záznamům vytváří standardní SRT soubor určený pro titulky. Ten v tomto případě místo titulků obsahuje aktuální rychlost, GPS souřadnice a název ulice v místě polohy vozidla. Jak přesně vypadá jeden takový záznam je vidět v úryvku:
#
# \begin{verbatim}
# 27
# 00:00:52,185 --> 00:00:53,185
# [2017-04-13 16:54:09]    52Km/h
# 49,949684,15,324726
# 38, 285 33 Církvice, Česko
#
# \end{verbatim}
#
# Pořizování videozáznamů probíhalo z osobní automobilu, kdy byl fotoaparát umístěn pomocí přísavného držáku na čelním skle, tak aby se dotýkal spodní části palubní desky automobilu a minimalizoval se tím vliv otřesů, které způsobují značné rozmazání obrazu.
#
# Nahrávání probíhalo na různých typech pozemních komunikací, tak aby bylo možné sesbírat co nejpestřejší škálu dopravních značek. Nicméně silnice druhých a třetích tříd se pro tyto účely ukázaly jako nevhodné vzhledem k nízké hustotě pokrytí dopravního značení. Nejvhodnější bylo pořizovat záznamy na silnicích prvních tříd, které slouží jako obchvat centra měst, zde je velmi často regulována maximální povolená rychlost. Díky tomu se povedlo sesbírat dostatečné množství značek rychlostního omezení nebo například značek zakazujících vjezd cyklistům či vstup chodcům. Naopak pro značení typu zákazu odbočení bylo nutné zajet do centra měst, kde je výskyt těchto typů častější.
#
# Většina sbírání probíhalo v zimních měsících, kdy jsou zhoršené světelné podmínky, které zhoršují kvalitu pořízeného záznamu, jelikož čočka mobilního telefonu nižšístřední třídy není schopna tyto podmínky, tak dobře zpracovat. To však může simulovat zhoršené podmínky i v jiném období jelikož navrhované postupy musí počítat s méně kontrastním zašedlým obrazem.
#
# \section{Dataset pro kaskádový klasifikátor}
# \label{sec:datasetDetector}
#
# Pro netrénování kaskádového klasifikátoru, tak aby byl schopen detekovat zákazové dopravní značky je zapotřebí připravit sadu pozitivních a negativních snímků. Pozitivní snímky jsou v tomto případě zákazové dopravní značky. Vzhledem k tomu, že množství pořízených videí je poměrně rozsáhlé, není vhodné vyhledávat zákazové dopravní značení ručním procházením.
#
# Z toho důvodu je pro zautomatizování procesu vytvořen podpůrný skript umožňující jednoduchou detekci pomocí segmentace v barevném modelu a následné detekce tvarů, v tomto případě kružnic. Vstupní obraz je interpretován ve standardním RGB modelu, ten však není vhodný pro vyhledávání na základě barvy \cite{mohdali2013}. Proto je obraz převeden do modelu HSV, který umožňuje vhodněji nastavit rozsah určující, že se jedná o požadovanou barvu. A to hlavně z toho důvodu, že konkrétní barva se sestává nejen z barevného tónu, ale také ze sytosti a úrovně jasu. To jsou atributy, které se ve snímcích v oblasti dopravní značky velmi mění a díky možnosti jejich nastavení je lze snadno odchytit.
#
# Z obrazu převedeném do modelu HSV je vytvořena černobílá reprezentace z oblastí, které vyhovují rozsahu se spodní hranicí \emph{150, 45, 45} a horní \emph{210, 255, 255}.  Pokud počet nenulových bodů v binární reprezentaci dosahuje experimentálně určené hodnoty, je snímek předán k dalšímu zpracování. V tom dojde k vyhledání geometrických tvarů, konkrétně kružnic a to pomocí Houghových transformací. Konkrétní parametry těchto kružnic je minimální poloměr $25\,px$ a maximální $50\,px$. Minimální vzdálenost středů kružnici je nastavena na $50\,px$ což zabraňuje vícenásobným detekcím stejného prostoru, ale zároveň dovoluje detekovat různé dopravní značky vedle sebe. Detekovaná oblast potom pravděpodobně odpovídá zákazové dopravní značce.
#
# Sesbírané záznamy jsou postupně procházeny a na každý pátý snímek je aplikován popsaný proces. V případě detekování objektu je výřez uložen pro další zpracování. Během zpracování těchto záznamů probíhá také sbírání negativních snímků. Jako negativní snímek je použit každý šedesátý vstupní snímek na kterém nejsou detekovány žádné dopravní značky.
#
# Soubory byly vyexportovány do dvou skupin u kterých bylo nutné provést ruční kontrolu. To znamenalo v případě pozitivních snímků odebrat všechny vzorky, které neodpovídají dopravní značce a přesunout je do kategorie negativních snímků. Negativní snímky byly také ručně zkontrolovány a případně došlo k odstranění těch, které obsahovali dopravní značku.
#
# Celkově se takto povedlo vytvořit sadu obsahující 975 pozitivních snímků. Negativních snímků bylo tímto postupem vytvořeno 4500, nicméně pro trénování se počítá s využitím přibližně dvojnásobku pozitivních snímků.
#
# \begin{table}[t]
# \centering
# \label{tab:datasetType}
# \begin{tabular}{rlrr}
# ID & Název                                 & Počet pro trénování & Počet pro testování \\
# \toprule
# 1  & Zákaz vstupu chodců                   & 7                   & 4                   \\
# 2  & Zákaz vjezdu jízdních kol             & 45                  & 14                  \\
# 3  & Zákaz vjezdu nákladních automobilů    & 130                 & 25                  \\
# 4  & Omezení rozměrů nákladních automobilů & 10                  & 5                   \\
# 5  & Zákaz předjíždění                     & 110                 & 15                  \\
# 6  & Nejvyšší povolená rychlost            & 574                 & 45                  \\
# 7  & Zákaz stání                           & 12                  & 6                   \\
# 8  & Zákaz odbočování vlevo                & 55                  & 14                  \\
# 9  & Zákaz odbočování vpravo               & 21                  & 8                   \\
# 10 & Zákaz zastavení                       & 151                 & 32                  \\
# 11 & Zákaz vjezdu                          & 12                  & 4                   \\
# 12 & Omezení rozměrů vozidla               & 29                  & 8                   \\
# 13 & Nejvyšší povolená hmotnost            & 30                  & 10
# \end{tabular}
# \caption{Datová sada pro trénování a testování rozpoznávání typu dopravních značek}
# \end{table}
#
# \subsection{Testovací sada}
# \label{sec:DatasetTest}
#
# Pro otestování klasifikátoru je zapotřebí připravit testovací sadu, pomocí které bude možné určit parametry úspěšnosti klasifikaci. Proto bylo natočeno 25 minut záznamů stejným mobilním telefonem jakým probíhalo vytváření datové sady a dalších 10 minut jiným mobilním telefonem. Všechny tyto záznamy pochází z jiných lokalit než záznamy pro trénování a zahrnují jak centra měst, tak i jejich periferie. Pro účely testování je z těchto záznamů vybrán každý čtyřicátý snímek, jejichž celkový počet odpovídá číslu 1570. Pro všechny tyto snímky jsou vytvořeny anotace pomocí připraveny nástroje, který slouží pro zakreslení úsečky od levého horního okraje k pravému spodnímu okraji značky. Počáteční a koncové souřadnice se uloží k názvu souboru, aby mohly být později využity při testování. Výsledkem je 92 anotovaných dopravních značek na 73 snímcích.
#
# \section{Dataset pro klasifikaci typu}
# \label{sec:TypeDataset}
#
# K natrénování klasifikátoru pro klasifikaci typu dopravní značky je také nutné sestavit datovou sadu. Ta vychází z té předcházející plus je rozšířena o nové vzorky, které jsou vybrány zpracováním dalších záznamů z kterých jsou tentokrát vybírány dopravní značky pomocí kaskadávého klasfikátor, který vykazuje značně menší míru \emph{False Positive}.
#
# Všechny vzorky je nutné roztřídit do jednotlivých tříd podle typu dopravní značky. To z počátku probíhalo výhradně ručně. Po naplnění tříd několika vzorky byl natrénován navrhnutý klasifikátor na této malé datové sadě. To však stačilo k tomu, aby byl proces třídění o něco jednodušší díky tomu, že soubory byly tříděny pomocí skriptu a klasifikátoru, jehož úspěšnost sice nebyla vysoká, ale značně zjedndoušila celý proces. Nad takto setříděnými vzorky už stačilo provést jednoduchou ruční korekci. Výsledná datová sada byla roztříděna opět na skupinu pro trénování a testování, konkrétní seznam typů dopravních značek a počet příslušných vzorků je vidět v tabulce ~\ref{tab:datasetType}.
#
#
#
# \section{Dataset pro klasifikaci hodnot}
# \label{sec:datasetValue}
#
# Aby bylo možné natrénovat klasifikátor hodnot rychlostního omezení, tak je zapotřebí připravit datovou sadu, která bude obsahovat všechny číselné symboly, které se na tomto typu dopravní značky objevují, tedy čísla 0-9. K tomu byly vybrány vzorky z předchozí sady ze skupiny rychlostního omezení, k těm byly doplněny další vzorky zpracováním nových záznamů pomocí implementovaného rozpoznávání typu dopravní značky, dohromady bylo takto sesbíráno 1005 vzorků. Z celkového počtu bylo odděleno 112 vzorků pro testovací sadu.
#
# K účelům trénování je nutné ze vzorků vyextrahovat jednotlivé symboly, k tomu slouží postupy navrhnuté v kapitole ~\ref{fig:ValueClassification}. Ty byly podobným postupem jako v předchozí části postupně roztříděny do oddělených skupin, počet vzorků v rámci skupiny znázorňuje tabulka ~\ref{tab:datasetValue}.
#
# \begin{table}[ht]
# \centering
# \label{tab:datasetValue}
# \begin{tabular}{rr}
# Symbol & Počet \\
# \toprule
# 0      & 1160  \\
# 1      & 51    \\
# 2      & 13    \\
# 3      & 93    \\
# 4      & 46    \\
# 5      & 232   \\
# 6      & 310   \\
# 7      & 151   \\
# 8      & 169   \\
# 9      & 22
# \end{tabular}
# \caption{Dataset pro trénování klasifikátoru symbolů}
# \end{table}
#
# \begin{table}[ht]
# \centering
# \label{tab:datasetValueTest}
# \begin{tabular}{rr}
# Hodnota & Celkem  \\
# \toprule
# 30 & 7	 \\
# 40 & 5 	\\
# 50 & 8 	\\
# 60 & 31	\\
# 70 & 14  \\
# 80 & 30  \\
# 90 & 4 \\
# 100 & 6  \\
# 130 & 1
# \end{tabular}
# \caption{Sada pro testování klasifikace rychlostního limitu}
# \end{table}
#
#
# Testovací sada se nesestává ze skupin samostatných symbolů, ale ze vzorků dopravních značek rychlostního omezení. To přináší možnost testovat nejen samotnou klasifikaci symbolů, ale testovat i úspěšnost segmentace bez které by klasifikace symbolů neměla žádný smysl. Značky jsou rozdělen do tříd podle jejich hodnoty jak znázorňuje tabulka ~\ref{tab:datasetValueTest}.
#
#
# %%%%%%%  IMPLEMENTACE  %%%%%%%
#
# \chapter{Implementace}
#
# Tato kapitola se věnuje konkrétní implementaci navrhnutých algoritmů a postupů v předchozích částech. Stěžejní částí této kapitoly je popis postupů při trénování kaskádové klasifikátoru pro detekování dopravních značek v obraze. To je následováno popisem trénování klasifikátoru pro rozpoznávání konkrétních druhů značek a klasifikace hodnot pro rychlostní omezení.
#
# Všechny vytvořené skripty v rámci  této práce jsou implementovány ve skriptovacím jazyce Python s podporou pro verze 3.5 a vyšší. Hlavní komponentou sloužící k realizaci potřebných algoritmů je knihovna pro podporu počítačového vidění OpenCV 3.
#
# \section{Trénování kaskádového klasifikátoru}
#
# Prvním krokem je vytvoření pozitivních vzorků z připravené datové sady v sekci ~\ref{sec:datasetDetector}.  K jejich vytváření je využívána utilita \emph{opencv\_createsamples}, která dokáže vzít množinu snímků a z nich podle připraveného anotačního souboru vytvořit vektor pozitivních vzorků.  Díky tomu, že v datové sadě jsou snímky obsahující pouze výřez jedné dopravní značky, není nutné speciálně vytvářet anotace s informací o pozici dopravní značky v rámci snímku. Anotace tedy odpovídá velikosti snímku s informací o tom, že je ve snímku přítomna pouze jedna dopravní značka.  Anotační soubor je vytvořen jednoduchým skriptem, který prochází zadanou složku a  načítá v ní všechny obrázky. Po načtení sestaví absolutní cestu ke snímku za kterou doplní jeho velikost a číslo jedna odpovídající počtu dopravních značek na snímku.
#
# Pro vytvoření pozitivních snímků je zvolena velikost $30\times30$\,px, celkově je použitou 975 vzorků viz. konkrétní příkaz pro spuštění:
#
# \begin{verbatim}
# opencv_createsamples -info info.dat -num 975 -w 30 -h 30 -vec samples.vec
# \end{verbatim}
#
# Pozitivní snímky z předchozího kroku jsou již připravený, dalším nutným předpokladem je sada negativních snímků jejichž počet odpovídá přibližně dvojnásobku pozitivních snímků a její příprava je popsána v  zde ~\ref{sec:datasetDetector}.  Pro spuštění je potřeba mít připravený soubor obsahující všechny negativní snímky, tentokrát však není nutné zapisovat doplňující informace o velikosti snímku, ale stačí pouhý seznam snímků s absolutní cestou k jednotlivým souborům.
#
# K trénování kaskádového klasifikátoru je využívána opět utilita knihovny OpenCV a to konkrétně \emph{opencv\_traincascaded}, ta narozdíl od starší \emph{opencv\_haartraining} umožňuje volbu příznaků LBP, které jsou v tomto případě použity. Parametr \emph{maxFalseAlarmRate} je snížen na hodnotu 0.4 pro dosažení nižšího počtu false positive detekcí. Pozitivních snímků je zadáno pouze 900, jelikož v každé další úrovni, kterých je celkově 18 se počet použitých pozitivních snímků zvyšuje.  Pozitivní snímky jsou zahrnuty v souboru \emph{samples.vec} a seznam negativních snímků je předáván prostřednictvím souboru \emph{bg.txt},  přesné spuštění odpovídá následujícímu příkazu:
#
# \begin{verbatim}
# opencv_traincascaded -data training -vec samples.vec -bg bg.txt
#     -numStages 10 -numPos 900 -numNeg 2000 -w 30 -h 30
#     -featureType LBP -maxFalseAlarmRate 0.4
# \end{verbatim}
#
#
# Proces trénování na počítači s procesorem Intel Core i3 4330 s frekvencí 3,4\, GHz trval 1 den a 16 hodin.
#
#
# \section{Trénování klasifikátoru pro klasifikaci typu}
#
# Za účelem natrénování klasifikátoru pro rozpoznání typu dopravní značky, který  využívá metodu k-Nearest Neighbor je použita datová sada připravená v předchozí kapitole ~\ref{sec:TypeDataset}.  K té je připraven soubor se seznam souborů a příslušnou informací o jaký typ se jedná.  Tento soubor je předán skriptu, který načítá postupně všechny vzorky. Nad každým se pokusí omezit výběr pouze na oblast dopravní značky a odstranit přebytečné okolí. Výběr je pak prahováním převeden na binární reprezentaci, která je zmenšena na velikost  $30\times30 \,px$, převedena do reprezentace jednorozměrného pole. Takto upravené pole je  uloženo do souboru pro vzorky, který využívá  klasifikátor pro určení třídy daného vstupního vzorku. Vedle toho je postupně vyplňován soubor, který obsahuje informace k jaké třídě takto vytvořené příznaky přísluší.
#
#
# \section{Trénování klasifikátoru hodnoty}
#
# Postup trénování klasifikátoru hodnoty rychlostního omezení je velmi podobný jelikož je zde opět pro klasifikaci použita metoda KNN.  Vstupem je tentokrát datová sada jejíž specifické vlastnosti jsou popsány v sekci ~\ref{sec:datasetValue}. Datová sada je již roztříděna do tříd a vytvořený soubor s odkazy na všechny vzorky obsahuje vždy i zadanou skupinu. Vstupní vzorky z datové sady tentokrát nejsou nijak výrazně upravovány jelikož už jsou při vytvoření pouze v černobílé  reprezentaci. Skript pro natrénování tedy opět postupně načítá všechny vzorky a jako v předchozím případě  změní jejich velikost, tentokrát jde o zmenšení na $10\times10\,px$, tato matice je opět převedena na jednodimenzionální podobu a serializována do souboru vstupních vzorků, který  jako v předchozím případě doplňuje druhý soubor s informacemi o příslušnosti  příznaků.
#
#
#
# %%%%%%%  VYHDONOCENÍ  %%%%%%%
# \chapter{Vyhodnocení}
#
# Vyhodnocení bylo rozděleno na samostatné části odpovídající návrhu řešení, tak aby bylo možné otestovat všechny komponenty samostatně.
#
# \section{Vyhodnocení kaskádového klasifikátoru jako detektoru dopravních značek}
#
# Tato sekce se věnuje experimentálnímu otestování kaskádového klasifikátoru jako detektoru zákozových dopravních značek. K testování je použita datová sada popsaná v sekci ~\ref{sec:DatasetTest}. Vyhodnocení je prováděno pomocí vytvořeného skriptu, který k detekování používá jako v samotné implementaci metodu \emph{detectMultiscale} a to i se stejnými parametry. Pro experimenty byly spuštěny celkem tři testy při kterých je experimentováno s hodnotou parametru \emph{scaleFactor} určující poměr zvětšení detekčního okna. Tabulka ~\ref{tab:detector-experiments} ukazuje vyhodnocení detektoru na čtyřech různých parametrech:
#
# \begin{itemize}
# \item True Positive (TP) - detekovaný výběr odpovídá dopravní značce
# \item False Positive (FP) - detekovaný výběr neodpovídá dopravní značce
# \item True Negative (TN) - na snímku není žádná značka a nebyl detekován žádný výběr
# \item False Negative (FN) - nebyla detekována přítomná dopravní značka
# \end{itemize}
#
# Všechny tyto parametry jsou vyjadřovány poměrově v procentech k příslušnému počtu snímků. Výsledky naznačují, že nejvhodnější volbou pro parametr \emph{scaleFactor} je hodnota $1,2$. Při nastavení nižší hodnoty se sice zvýší úspěšnost true positive detekce, ale zvýší se také hodnota false positive, kterou chceme udržet na co nejnižší úrovni. Shrnující výsledky jsou zobrazeny v tabulce ~\ref{tab:detector-experiments}. Při bližším zkoumání bylo zjištěno, že největšímu počtu false negative detekcí dochází v případě výskytu dopravní značky typu zákazu zastavení nebo zákazu stání. To je pravděpodobně způsobeno jejich nízkým zastoupením v trénovací sadě a odlišnou charakteristikou na rozdíl od jiných dopravních značek jako je rychlostní omezení nebo zákaz předjíždění.
#
# \begin{table}[H]
# \centering
# \label{tab:detector-experiments}
# \begin{tabular}{rrrrr}
# Scale factor & TP [\%]    & FP [\%]   & FN [\%]   & TN [\%]    \\
# \toprule
# 1,1          & 91.30 & 2.42 & 8.70  & 97.80 \\
# 1,2          & 89.13 & 0.76 & 10.87 & 99.20 \\
# 1,3          & 85.87 & 0.70 & 14.13 & 99.33
# \end{tabular}
# \caption{Procentuální vyhodnocení detektoru zákazových dopravních značek}
# \end{table}
#
# Pro lepší vyjádření úspěšnosti detektoru je vhodné znázornit na grafu parametr TRP, tedy počet true positive detekcí ku celkovému počtu dopravních značek v závislosti na FRP, což je hodnota odpovídající počtu false positive detekci ku celkovému počtu snímků. Výsledek vyjadřuje ROC křivka v grafu ~\ref{tab:detector-experiments}. Je tak možné vidět, že detektor je velmi kvalitní, jelikož křivka ukazuje velmi strmý růst, který vyjadřuje, že při detekování dochází pouze k minimální počtu false positive detekcí.
#
#
#
# \begin{figure}[h]
# \centering
# \includegraphics[width=0.8\linewidth,]{obrazky-figures/ROC.pdf}\\[1pt]
# \caption{ROC křivka}
# \label{fig:Symbols}
# \end{figure}
#
#
# \section{Výsledky klasifikace typu dopravní značky}
#
# Pro vyhodnocování rozlišení typu dopravního značení je využívána připravená datová sada popsána v sekci ~\ref{sec:TypeDataset}, ta obsahuje výřezy dopravních značek rozdělených do několika skupin. Při vyhodnocování úspěšnosti probíhá postup oříznutí výběru a vyextrahování příznaků navrhnutý v sekci ~\ref{sec:TypeDesign}. Spuštění skriptu pro testování provede postupné klasifikování všech značek v trénovací sadě a porovnání se zadanými anotacemi. Celkové výsledky úspěšnosti ukazuje tabulka ~\ref{tab:type-experiments-summary}.
#
# \begin{table}[ht]
# \centering
# \label{tab:type-experiments-summary}
# \begin{tabular}{rrrrr}
# Celkem   & TP	&	FP	 & Úspěšnost    \\
# \toprule
# 190 & 168 & 22 & 88.42\,\%
# \end{tabular}
# \caption{Celková úspěšnost klasifikace typu dopravních značek}
# \end{table}
#
# Výsledná hodnota představuje poměrně dobrý výsledek, nicméně může skrývat některé detaily způsobené nevyvážeností sestavené testovací sady. Proto byla sestavena tabulka ~\ref{tab:type-experiments}, která znázorňuje úspěšnost a počty úspěšných detekcí. Z uvedených výsledků je vidět, že nejproblematičtějším typem je dopravní značka \emph{omezení rozměrů nákládních automobilů}. To je způsobeno tím, že trénovací sada obsahuje příliš málo vzorků této značky. Navíc velmi vysoká podobnost se zákazem vjezdu nákladních automobilů způsobí absolutní záměnu právě s tímto typem, což ilustruje tabulka ~\ref{tab:type-confusion-1} představující matici záměn klasifikace jednotlivých typů dopravního značení.
#
# \begin{table}[H]
# \centering
# \label{tab:type-experiments}
# \begin{tabular}{rlrrr}
# ID & Zkratka & Celkem   & TP    & Úspěšnost    \\
# \toprule
# 1 & Zákaz vstupu chodců & 4 & 3 & 75.00\,\%  \\
# 2 & Zákaz vjezdu jízdních kol & 14 & 12 & 85.71\,\%  \\
# 3 & Zákaz vjezdu nákladních automobilů & 25 & 25 & 100.00\,\%  \\
# 4 & Omezení rozměrů nákladních automobilů & 5 & 0 & 0.00\,\%  \\
# 5 & Zákaz předjíždění & 15 & 13 & 86.67\,\%  \\
# 6 & Nejvyšší povolená rychlost & 45 & 44 & 97.78\,\%  \\
# 7 & Zákaz stání & 6 & 1 & 16.67\,\%  \\
# 8 & Zákaz odbočování vlevo & 14 & 14 & 100.00\,\%  \\
# 9 & Zákaz odbočování vpravo & 8 & 7 & 87.50\,\%  \\
# 10 & Zákaz zastavení & 32 & 31 & 96.88\,\%  \\
# 11 & Zákaz vjezdu & 4 & 4 & 100.00\,\%  \\
# 12 & Omezení rozměrů & 8 & 7 & 87.50\,\%  \\
# 13 & Nejvyšší povolená hmotnost & 10 & 7 & 70.00\,\%
#
# \end{tabular}
# \caption{Úspěšnost klasifikace jednotlivých tříd}
# \end{table}
#
# Aby bylo možné omezit neplatné klasifikace je nutné stanovit maximální hranici tolerance rozpoznání typu. V případě klasfikace pomocí metody k-Nearest Neigbor toho lze dosáhnout tím, že experimentálně určíme nad jakou hranici vzdálenosti porovnávaného vzorku s testovacími příznaky rozhodneme, že je pro nás výsledek klasifikace neplatný. V tomto případě je provedeno další měření, kdy je jako tento limit stanovena hodnota $9000000$. Výsledkem je znatelně nižší celková úspěšnost detekce, která je nyní pouhých $72,10\,\%$ a u některých typů dopravního značení jako je například zákaz zastavení jak je vidět v tabulce ~\ref{tab:type-experiments-2} došlo k výraznému poklesu úspěšnosti.
#
#
# \begin{table}[t]
# \centering
# \label{tab:type-experiments-2}
# \begin{tabular}{rlrrr}
# ID & Zkratka & Celkem   & TP    & Úspěšnost    \\
# \toprule
# 1 & Zákaz vstupu chodců & 4 & 3 & 75.00\,\%  \\
# 2 & Zákaz vjezdu jízdních kol & 14 & 12 & 85.71\,\%  \\
# 3 & Zákaz vjezdu nákladních automobilů & 25 & 23 & 92.00\,\%  \\
# 4 & Omezení rozměrů nákladních automobilů & 5 & 0 & 0.00\,\%  \\
# 5 & Zákaz předjíždění & 15 & 12 & 80.00\,\%  \\
# 6 & Nejvyšší povolená rychlost & 45 & 39 & 86.67\,\%  \\
# 7 & Zákaz stání & 6 & 1 & 16.67\,\%  \\
# 8 & Zákaz odbočování vlevo & 14 & 11 & 78.57\,\%  \\
# 9 & Zákaz odbočování vpravo & 8 & 3 & 37.50\,\%  \\
# 10 & Zákaz zastavení & 32 & 16 & 50.00\,\%  \\
# 11 & Zákaz vjezdu & 4 & 4 & 100.00\,\%  \\
# 12 & Omezení rozměrů & 8 & 7 & 87.50\,\%  \\
# 13 & Nejvyšší povolená hmotnost & 10 & 6 & 60.00\,\%
# \end{tabular}
# \caption{Úspěšnost klasifikace jednotlivých tříd při zavedení maximální přípustné vzdálenosti příznaků}
# \end{table}
#
#
# \begin{table}[H]
# \centering
# \label{tab:type-confusion-1}
# \begin{tabular}{c|ccccccccccccc}
#  & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 \\
# \toprule
#
# 1 & \textbf{3} & 0 & 0 & 0 & 0 & \textbf{1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 2 & 0 & \textbf{12} & 0 & 0 & \textbf{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 3 & 0 & 0 & \textbf{25} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 4 & 0 & 0 & \textbf{5} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 5 & 0 & 0 & 0 & 0 & \textbf{13} & \textbf{1} & 0 & 0 & 0 & \textbf{1} & 0 & 0 & 0 \\
# 6 & 0 & 0 & 0 & 0 & 0 & \textbf{44} & 0 & 0 & 0 & \textbf{1} & 0 & 0 & 0 \\
# 7 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & \textbf{2} & \textbf{1} & \textbf{2} & 0 & 0 & 0 \\
# 8 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{14} & 0 & 0 & 0 & 0 & 0 \\
# 9 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & \textbf{7} & 0 & 0 & 0 & 0 \\
# 10 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & 0 & 0 & \textbf{31} & 0 & 0 & 0 \\
# 11 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{4} & 0 & 0 \\
# 12 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & 0 & 0 & 0 & 0 & 0 & \textbf{7} & 0 \\
# 13 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{3} & 0 & \textbf{7}
# \end{tabular}
# \caption{Matice záměn tříd}
# \end{table}
#
# Nižší úspěšnost je vyvážena snížením počtu záměn a tím pádem zvýšení spolehlivosti v rozpoznání typu, záleží tedy jaké je výsledné použití. V případě této práce je pro zpracování záznamu implementována první varianta, jelikož se při zpracování klasifikuje poslední výskyt dopravní značky a je porovnán s předcházejícími snímky, díky tomu dojde k odfiltrování nevalidní klasifikace.
#
# \begin{table}[H]
# \centering
# \label{tab:type-confusion-2}
# \begin{tabular}{c|ccccccccccccc}
#  & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 \\
# \toprule
#
# 1 & \textbf{3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 2 & 0 & \textbf{12} & 0 & 0 & \textbf{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 3 & 0 & 0 & \textbf{23} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 4 & 0 & 0 & \textbf{5} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 5 & 0 & 0 & 0 & 0 & \textbf{12} & 0 & 0 & 0 & 0 & \textbf{1} & 0 & 0 & 0 \\
# 6 & 0 & 0 & 0 & 0 & 0 & \textbf{39} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
# 7 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & 0 & 0 & \textbf{1} & 0 & 0 & 0 \\
# 8 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{11} & 0 & 0 & 0 & 0 & 0 \\
# 9 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{1} & \textbf{3} & 0 & 0 & 0 & 0 \\
# 10 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{16} & 0 & 0 & 0 \\
# 11 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{4} & 0 & 0 \\
# 12 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{7} & 0 \\
# 13 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \textbf{3} & 0 & \textbf{6}
# \end{tabular}
# \caption{Matice záměn tříd při nastavení omezení}
# \end{table}
#
#
# \section{Klasifikace rychlostního limit}
#
# K vyhodnocení úspěšnosti klasifikace hodnoty rychlostního omezení je tentokrát použit trochu jiný typ datové sady. Ta se nesestává ze samotných symbolů, jelikož proces klasifikace hodnoty zahrnuje nejprve segmentaci symbolů a až po té jejich postupnou klasifikaci. Za účelem testování touto metodikou je připravena datová sada s rozložením viz. tabulka ~\ref{tab:datasetValue}.
#
# Vyhodnocovací skript pracuje s připravenou sadou, kdy na každou značku aplikuje postup segmentace znaků ~\ref{sec:ValueDesign}. Následně klasifikuje samostatné znaky a pokouší se sestavit výslednou hodnotu. Celkový výsledek úspěšnosti je znázorněn tabulce ~\ref{tab:value-experiments-summary}.
#
# \begin{table}[ht]
# \centering
# \label{tab:value-experiments-summary}
# \begin{tabular}{rrrrr}
# Celkem   & TP	&	FP	 & Úspěšnost    \\
# \toprule
# 106 & 97 & 9 & 91.51\,\%
# \end{tabular}
# \caption{Celková úspěšnost klasifikace hodnoty rychlostního limitu}
# \end{table}
#
#
# Výsledná úspěšnost je zde velmi dobrá a to i při pohledu na samostatné třídy hodnot v tabulce ~\ref{tab:value-experiments-classess}. Často se objevující rychlostní omezení jsou klasifikovány velmi přesně díky velkému zastoupení symbolů v trénovací sadě. Největší problém se objevuje u klasifikace rychlostního limitu s hodnotou 100. To je způsobeno nedokonalou segmentací jednotlivých symbolů v oblasti dopravní značky. V testovacích datech se v tomto případě neobjevují snímky s ideálními podmínkami a segmentace, tak selhává při menších rozestupech jednotlivých symbolů.
#
# \begin{table}[t]
# \centering
# \label{tab:value-experiments-classess}
# \begin{tabular}{rrrr}
# Hodnota & Celkem   & TP    & Úspěšnost    \\
# \toprule
# 30 & 7 & 7 & 100.00\,\%  \\
# 40 & 5 & 5 & 100.00\,\%  \\
# 50 & 8 & 8 & 100.00\,\%  \\
# 60 & 31 & 31 & 100.00\,\%  \\
# 70 & 14 & 11 & 78.57\,\%  \\
# 80 & 30 & 29 & 96.67\,\%  \\
# 90 & 4 & 3 & 75.00\,\%  \\
# 100 & 6 & 2 & 33.33\,\%  \\
# 130 & 1 & 1 & 100.00\,\%
#
# \end{tabular}
# \caption{Úspěšnost klasifikace hodnot podle připravených skupin}
# \end{table}
#
# \section{Celkové vyhodnocení při zpracování jednoho záznamu}
#
# \todo{Ukázat jak moc to funguje, když to dělá vše dokupy a co z toho umí vypadnout}
#
#
# \chapter{Závěr}
#
# V práci je popsán proces vytváření potřebných datových sad. Pro práci s kaskádovým klasifikátorem je připravena datová sada zahrnující 975 pozitivních vzorků. Pro účely testování kaskádového klasifikátoru vzniklo 30 minut záznamů z různého prostředí nad jehož snímky byl detektor testován a jeho úspěšnost dosahuje téměř 90\,\%.  Datová sada připravená pro trénování a testování klasifikátoru typu dopravní značky je problematická v nerovnoměrném vyvážení výskytu různých typů, což je dáno jejich charakteristikou výskytu, výsledná úspěšnost klasifikace odpovídá 88\,\%.  Datová sada  připravená pro klasifikaci hodnot se potýká s podobným problémem, ale  díky standardizovaným vlastnostem symbolů vykazuje velmi vysokou úspěšnost klasifikace  91\,\%, která je z  největší části ovlivněna úspěchem segmentace symbolů na dopravní značce.
#
# Během práce bylo zjištěno, že základní metoda klasifikace k-Nearest Neigbour nemusí být nejvhodnější volbou pro klasifikaci typu  dopravní značky a  mohlo by být zajímavější pokusit  se o využití sofistikovanějších metod jako jsou například konvoluční neuronové sítě.  Jako další  vývoj je možnost zapracovat určení pozice v rámci prostoru a určit tak například pro který pruh nebo směr má značka platnost.
#
# Povinně se zde objeví i zhodnocení z pohledu dalšího vývoje projektu, student uvede náměty vycházející ze zkušeností s řešeným projektem a uvede rovněž návaznosti na právě dokončené projekty.
#
# %=========================================================================
#

